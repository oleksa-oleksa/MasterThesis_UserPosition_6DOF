import torch
import torch.nn as nn
import logging
import os
import math


class LSTMFCNModel1(nn.Module):
    """
        Implements MLSTM FCN model from the paper
        MLSTM Fully Convolutional Networks for Time Series Classification,
        augment the fast classification performance of Temporal Convolutional
        layers with the precise classification
        of Long Short Term Memory Recurrent Neural Networks.

        The input to the model is of the shape (Batchsize, Number of timesteps, Number of variables)

        The shuffle operation is applied before the LSTM to obtain the input shape
        (Batchsize, Number of variables, Number of timesteps).

        The original shape (Batchsize, Number of timesteps, Number of variables)
        is passed to Fully Convolutional Network (FCN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Works both on CPU and GPU without additional modifications"""
        super(LSTMFCNModel1, self).__init__()
        self.name = "LSTM-FCN1"

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim_lstm = hidden_dim
        self.layer_dim = 1
        self.dropout = 0
        self.dropout_layer = 0.8
        self.timestamps = 20

        '''
        original MLSTM-FCN from work of Karim et al. has Masking layer in Keras
        # Masking is a way to tell sequence-processing layers that certain timesteps in an input are missing, 
        and thus should be skipped when processing the data
        We assume to get the dataset without missed timestemps because HoloLens is able 
        to deliver sensor measurements constantly. 
        '''
        # ========= LSTM with Dropout ===================== #
        # shape [batch_size, seq_len, input_size]

        self.lstm_model = nn.LSTM(input_size=self.timestamps, hidden_size=self.hidden_dim_lstm,
                            num_layers=self.layer_dim, batch_first=True, dropout=self.dropout)
        self.lstm_dropout = nn.Dropout2d(self.dropout_layer)
        # Fully connected layer maps last LSTM output (hidden dimension) to the label dimension
        self.lstm_fc = nn.Linear(self.hidden_dim_lstm, output_dim)

        # ========= Fully connected networrk  ===================== #
        self.conv1 = nn.Conv1d(in_channels=self.timestamps, out_channels=128, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(num_features=128, eps=1e-3, momentum=0.99)
        self.relu1 = nn.LeakyReLU()
        self.squeeze_1 = FCNSqueezeBLock(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(num_features=256, eps=1e-3, momentum=0.99)
        self.relu2 = nn.LeakyReLU()
        self.squeeze_2 = FCNSqueezeBLock(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(num_features=128, eps=1e-3, momentum=0.99)
        self.relu3 = nn.LeakyReLU()

        self.pool = nn.AdaptiveMaxPool1d(output_size=output_dim)

        self.fc = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax()

        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Model {self.name} on GPU with cuda: {self.cuda}")

    def pre_shuffle_input_lstm(self, x):
        """
        input will be pre-shuffled to be in the shape
        (Batchsize, Number of variables, Number of timesteps)
        """
        if self.cuda:
            x = x.cuda()
        return torch.permute(x, (0, 2, 1))

    def convert_to_cuda(self):
        self.lstm_model.cuda()
        self.lstm_dropout.cuda()
        self.lstm_fc.cuda()
        self.conv1.cuda()
        self.bn1.cuda()
        self.relu1.cuda()
        self.conv2.cuda()
        self.bn2.cuda()
        self.relu2.cuda()
        self.conv3.cuda()
        self.bn3.cuda()
        self.relu3.cuda()
        self.pool.cuda()
        self.fc.cuda()
        self.softmax.cuda()

    def forward(self, x):

        # ========= LSTM with Dropout ===================== #
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim_lstm).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim_lstm).requires_grad_()

        if self.cuda:
            x = x.cuda()
            h0, c0 = h0.cuda(), c0.cuda()

        print(f"lstm input: {x.size()}")
        # shuffle operation is applied before the LSTM to obtain the input shape
        # (Batchsize, Number of variables, Number of timesteps)
        x_shuffled = self.pre_shuffle_input_lstm(x)
        print(f"x_shuffled: {x_shuffled.size()}")

        x_lstm, (hn, cn) = self.lstm_model(x_shuffled, (h0.detach(), c0.detach()))
        print(f'ltsm out: {x_lstm.size()}')
        x_lstm = self.lstm_dropout(x_lstm)
        x_lstm_out = self.lstm_fc(x_lstm)
        print(f'lstm out fc: {x_lstm_out.size()}')

        # ========= Fully connected networrk  ===================== #
        # The original shape (Batchsize, Number of timesteps, Number of variables)
        # is passed to Fully Convolutional Network (FCN
        print(f"Input FCN: {x.size()}")
        x_fcn = self.conv1(x)
        x_fcn = self.bn1(x_fcn)
        x_fcn = self.relu1(x_fcn)
        print(f"FCN out 1 block 128: {x.size()}")
        x_fcn = self.squeeze_1(x_fcn)
        print(f"FCN squeeze 1: {x.size()}")


        x_fcn = self.conv2(x_fcn)
        x_fcn = self.bn2(x_fcn)
        x_fcn = self.relu2(x_fcn)
        x_fcn = self.squeeze_2(x_fcn)

        x_fcn = self.conv3(x_fcn)
        x_fcn = self.bn3(x_fcn)
        x_fcn = self.relu3(x_fcn)
        print(f'fcn before pooling: {x_fcn.size()}')
        x_fcn = self.pool(x_fcn)
        print(f'fcn after pooling: {x_fcn.size()}')

        out = torch.cat((x_lstm, x_fcn))

        out = self.fc(out)
        print(f'fcn after FC: {x_fcn.size()}')

        out = self.softmax(out)
        return out


class FCNSqueezeBLock(nn.Module):
    def __init__(self, filters):
        """
        Model is augmented by squeeze-and-excitation block to further improve accuracy.
        Uses a global average pool to generate channel-wise statistics
        The aggregated information obtained from the squeeze operation is followed by an excite operation,
        whose objective is to capture the channel-wise dependencies.
        To achieve this, a simple gating mechanism is applied with a sigmoid activation
        """
        super(FCNSqueezeBLock, self).__init__()
        self.name = "FCN Squeeze BLock"
        # squeeze block
        self.reduction_ratio = 16
        self.filters = filters
        self.reduced_filters = math.floor(filters / self.reduction_ratio)

        self.se_1 = nn.Linear(filters, self.reduced_filters)
        self.se_2 = nn.ReLU()
        self.se_3 = nn.Linear(self.reduced_filters, filters)
        self.se_4 = nn.Sigmoid()

        self.cuda = torch.cuda.is_available()

    def forward(self, x):
        if self.cuda:
            x = x.cuda()

        x = self.se_1(x)
        x = self.se_2(x)
        x = self.se_3(x)
        x = self.se_4(x)
        x = torch.matmul(self.filters, x)

        return x

