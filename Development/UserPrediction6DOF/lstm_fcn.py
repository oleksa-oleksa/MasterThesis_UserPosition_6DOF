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
        self.conv1 = nn.Conv1d(input_dim, 128, 8, padding='same', bias=False, stride=1)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, 5, padding='same', bias=False, stride=1)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, 3, padding='same', bias=False, stride=1)
        self.relu3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(128)

        self.glob_pool = nn.AvgPool1d(3)
        self.fc = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax()

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
        logging.info(F"Model {self.name} on GPU with cuda: {self.cuda}")

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        if self.cuda:
            x = x.cuda()

        # 2D LSTM

        x_lstm = self.lstm(x)
        x_lstm = self.dropout(x_lstm)

        # 1D FCN
        print(x.shape)
        x_fcn = torch.permute(x, (2, 1))

        x_fcn = self.conv1(x_fcn)
        x_fcn = self.relu1(x_fcn)
        x_fcn = self.bn1(x_fcn)

        x_fcn = self.conv2(x_fcn)
        x_fcn = self.relu2(x_fcn)
        x_fcn = self.bn2(x_fcn)

        x_fcn = self.conv3(x_fcn)
        x_fcn = self.relu3(x_fcn)
        x_fcn = self.bn3(x_fcn)

        x_fcn = self.glob_pool(x_fcn)

        out = torch.cat((x_lstm, x_fcn))
        out = self.fc(out)
        out = self.softmax(out)

        return out
