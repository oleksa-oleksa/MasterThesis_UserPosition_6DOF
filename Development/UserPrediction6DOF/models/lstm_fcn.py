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
        self.dropout_lstm = 0
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

        self.lstm = nn.LSTM(input_size=self.timestamps, hidden_size=self.hidden_dim_lstm,
                            num_layers=self.layer_dim, batch_first=True, dropout=self.dropout_lstm)
        self.dropout_lstm = nn.Dropout2d(self.dropout_layer)
        # Fully connected layer maps last LSTM output (hidden dimension) to the label dimension
        self.fc_lstm = nn.Linear(self.hidden_dim_lstm, output_dim)

        # ========= Fully connected networrk  ===================== #
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=8)
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
        self.lstm.cuda()
        self.dropout.cuda()
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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        if self.cuda:
            x = x.cuda()
            h0, c0 = h0.cuda(), c0.cuda()

        print(f"Input: {x.size()}")

        # shuffle operation is applied before the LSTM to obtain the input shape
        # (Batchsize, Number of variables, Number of timesteps)
        x_shuffled = self.pre_shuffle_input_lstm(x)
        print(f"x_shuffled: {x_shuffled.size()}")

        x_lstm, (hn, cn) = self.lstm(x_shuffled, (h0.detach(), c0.detach()))
        print(f'lstm after ltsm(x): {x_lstm.size()}')
        x_lstm = self.dropout(x_lstm)

        # 1D FCN
        # x_fcn = torch.squeeze(x)
        # print(f"x_fcn squeeze: {x_fcn.size()}")
        # dims in permute (tuple of python:ints) – The desired ordering of dimensions
        '''
        What permute function does is rearranges the original tensor according to the desired ordering, 
        note permute is different from reshape function, because when apply permute, 
        the elements in tensor follow the index you provide where in reshape it's not
        '''
        x_fcn = torch.permute(x, (0, 2, 1))
        print(f'fcn after permute: {x_fcn.size()}')

        x_fcn = self.conv1(x_fcn)
        x_fcn = self.bn1(x_fcn)
        x_fcn = self.relu1(x_fcn)
        x_fcn = self.squeeze_1(x_fcn)

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


class LSTMFCNModelSWP(nn.Module):
    """
        Implements LSTM FCN models, from the paper
        LSTM Fully Convolutional Networks for Time Series Classification,
        augment the fast classification performance of Temporal Convolutional
        layers with the precise classification
        of Long Short Term Memory Recurrent Neural Networks.

        The dimension shuffle transposes the input univariate time series of
        N time steps and1variable into a multivariate time series
        of N variables and 1 time step. In other words, when dimension shuffle
        is  applied  to  the  input  before  the  LSTM  block,  the LSTM block
        will process only1time step with N variables.

    """
    def __init__(self, seq_length_input, input_dim, hidden_dim, seq_length_output, output_dim, dropout, layer_dim):
        """Works both on CPU and GPU without additional modifications"""
        super(LSTMFCNModel, self).__init__()
        self.name = "LSTM-FCN"

        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.seq_length_input = seq_length_input  # sequence length
        self.seq_length_output = seq_length_output  # otput length of timeseries in the future
        self.N_time1 = batch_size
        self.layer_dim = layer_dim
        self.N_LSTM_Out = 128
        self.Conv1_NF = 128
        self.Conv2_NF = 256
        self.Conv3_NF = 128
        self.lstm_dropout = 0.2
        self.fcn_dropout = 0.3

        if 'FCN_PARAMETERS' in os.environ:
            self.fcn_dropout = float(os.getenv('FCN_DROPOUT'))
            self.lstm_dropout = float(os.getenv('LSTM_DROPOUT'))

        # LSTM layers (default 1)
        # setting batch_first=True requires the input to have the shape [batch_size, seq_len, input_size]
        self.lstm = LSTMModel1(self.input_dim, hidden_dim, self.N_LSTM_Out, dropout, layer_dim)

        self.C1 = nn.Conv1d(self.input_dim, 128, 1)
        self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 1)
        self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 1)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(p=self.lstm_dropout)
        self.ConvDrop = nn.Dropout(p=self.fcn_dropout)
        self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out, self.output_dim)

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            pass
        logging.info(F"Model {self.name} on GPU with cuda: {self.cuda}")

    def init_hidden(self):
        h0 = torch.zeros(self.layer_dim, self.N_time, self.N_LSTM_Out)  # .to(device)
        c0 = torch.zeros(self.layer_dim, self.N_time, self.N_LSTM_Out)  # .to(device)
        return h0, c0

    def forward(self, x):
        batch_size, sequence_length = x.shape[0], x.shape[1]

        # Initializing hidden state for first input with zeros
        if self.cuda:
            x = x.cuda()
        # print(f'x input: {x.size()}')

        # h0, c0 = self.init_hidden()
        # x1, (ht, ct) = self.lstm(x, (h0, c0))
        x1_lstm = self.lstm(x)
        # x1 = x1[:, -1, :]

        # print(f'x1_lstm: {x1_lstm.size()}')
        x2 = x.transpose(2, 1)
        # x2 = x
        # print(f'X2: {x2.size()}')
        # is [8192, 1, 10]
        # must [128, 10, 8]
        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1_lstm, x2), dim=1)
        x_out = self.FC(x_all)
        x_out = x_out.view([batch_size, -1, self.output_dim])
        return x_out


class LSTMFCNModelOld(nn.Module):
    """
        Implements LSTM FCN models, from the paper
        LSTM Fully Convolutional Networks for Time Series Classification,
        augment the fast classification performance of Temporal Convolutional
        layers with the precise classification
        of Long Short Term Memory Recurrent Neural Networks.

        The dimension shuffle transposes the input univariate time series of
        N time steps and1variable into a multivariate time series
        of N variables and 1 time step. In other words, when dimension shuffle
        is  applied  to  the  input  before  the  LSTM  block,  the LSTM block
        will process only1time step with N variables.

    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, layer_dim=1):
        """Works both on CPU and GPU without additional modifications"""
        super(LSTMFCNModelOld, self).__init__()
        self.name = "LSTM-FCN"

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers (default 1)
        # setting batch_first=True requires the input to have the shape [batch_size, seq_len, input_size]
        self.lstm = LSTMModel1(input_dim, hidden_dim, output_dim, dropout, layer_dim)

        if 'FCN_PARAMETERS' in os.environ:
            self.dropout = nn.Dropout2d(float(os.getenv('FCN_DROPOUT')))
        else:
            self.dropout = nn.Dropout2d(0.6)

        # PyTorch initializes the conv and linear weights with kaiming_uniform
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=8, padding='valid', bias=False, stride=1)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, 5, padding='valid', bias=False, stride=1)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, 3, padding='valid', bias=False, stride=1)
        self.relu3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(128)

        self.fc = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax()
        self.glob_pool = nn.AvgPool1d(3)


        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.lstm.cuda()
            self.dropout.cuda()
            self.conv1.cuda()
            self.relu1.cuda()
            self.bn1.cuda()
            self.conv2.cuda()
            self.relu2.cuda()
            self.bn2.cuda()
            self.conv3.cuda()
            self.relu3.cuda()
            self.bn3.cuda()
            self.glob_pool.cuda()
            self.fc.cuda()
            self.softmax.cuda()
        logging.info(F"Model {self.name} on GPU with cuda: {self.cuda}")

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        if self.cuda:
            x = x.cuda()

        # 2D LSTM

        print(f"x input: {x.size()}")
        x_lstm = self.lstm(x)
        print(f'lstm after ltsm(x): {x_lstm.size()}')
        x_lstm = self.dropout(x_lstm)

        # 1D FCN

        # x_fcn = torch.squeeze(x)
        # print(f"x_fcn squeeze: {x_fcn.size()}")
        # dims in permute (tuple of python:ints) – The desired ordering of dimensions
        # x_fcn = torch.permute(x_fcn, (1, 0))
        x_fcn = torch.permute(x, (1, 2, 0))

        print(f'fcn after permute: {x_fcn.size()}')

        x_fcn = self.conv1(x_fcn)
        x_fcn = self.relu1(x_fcn)
        x_fcn = self.bn1(x_fcn)

        x_fcn = self.conv2(x_fcn)
        x_fcn = self.relu2(x_fcn)
        x_fcn = self.bn2(x_fcn)

        x_fcn = self.conv3(x_fcn)
        x_fcn = self.relu3(x_fcn)
        x_fcn = self.bn3(x_fcn)

        x_fcn = self.fc(x_fcn)
        print(f'fcn after linear: {x_fcn.size()}')

        x_fcn = self.glob_pool(x_fcn)
        print(f'fcn after global pooling: {x_fcn.size()}')

        # x_fcn = torch.squeeze(x_fcn)
        # print(f"x_fcn squeeze: {x_fcn.size()}")

        x_fcn = self.softmax(x_fcn)
        print(f'fcn after softmax: {x_fcn.size()}')

        print(f'lstm before concat ltsm(x): {x_lstm.size()}')

        out = torch.cat((x_lstm, x_fcn))
        out = self.fc(out)
        out = self.softmax(out)

        return out
