import torch
import torch.nn as nn
import logging
import os
from .lstm import LSTMModel


class LSTMFCNModel(nn.Module):
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
        super(LSTMFCNModel, self).__init__()
        self.name = "LSTM-FCN"

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers (default 1)
        # setting batch_first=True requires the input to have the shape [batch_size, seq_len, input_size]
        self.lstm = LSTMModel(input_dim, hidden_dim, output_dim, dropout, layer_dim)

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
