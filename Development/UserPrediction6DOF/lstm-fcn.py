import torch
import torch.nn as nn
import logging
import os
from .lstm import LSTMModel


class LSTMFCNModel(nn.Module):
    """
        Implements a sequential network named Long Short Term Memory Network.
        It is capable of learning order dependence in sequence prediction problems and
        can handle the vanishing gradient problem faced by RNN.

        At each iteration, the LSTM uses gating mechanism and works with:
        1. the current input data Xt
        2. the hidden state aka the short-term memory Ht_1
        3. lastly the long-term memory Ct

        Three gates regulate the information to be kept or discarded at each time step
        before passing on the long-term and short-term information to the next cell.

        1. Input gate uses two layers:
            First layer I_1 selects what information can pass through it and what information to be discarded.
            Short-term memory and current input are passed into a sigmoid function. The output values
            are between 0 and 1, with 0 indicating unimportant data
            and 1 indicates that the information will be used.

                input_layer1 = sigma(W_input1 * (Ht_1, Xt) + bias_input1)

            Second layer takes same input information and passes it throught
            tanh activation function with outputs values between -1.0 and 1.0.

                input_layer2 = tanh(W_input2 * (Ht_1, Xt) + bias_input2))

            I_input is the information to be kept in the long-term memory and used as the output.

                input_gate = input_layer1 * input_layer2

        2. Forget gate keeps or discards information from the long-term memory.
            First short-term memory and current input is passed through a sigmoid function again

                forget_gate = sigma(W_forget * (Ht_1, Xt) + bias_forget)

            New long-term memory is created from outputs of the Input gate and the Forget gate

                Ct = Ct_1 * forget_gate + input_gate

        3. Output gate uses two layers:
            Gate takes the current input Xt, the previous short-term memory Ht_1 (hidden state)
            and long-term memory Ct computed in current step and ouputs the new hidden state Ht

            First layer short-term memory and current input is passed through a sigmoid function again

                output_layer1 = sigma(W_output1 * (Ht_1, Xt) + bias_output1)

            Second layer takes computed in current step long-term memory Ct and passes it
            with it owns weights througt tahn activation function

                output_layer2 = tanh(W_output2 * Ct + bias_output2))

            New hidden state is the output of output gate:

                output_gate = output_layer1 * output_layer2

                Ht = output_gate

        Short-term Ht and long-term memory Ct created by three gates will be passed over
        to the next iteration and the whole process will be repeated.

        The output on each iteration can be accessed through hidden state Ht

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
        self.conv1 = nn.Conv1d(input_dim, 128, 8, padding='same', bias=False)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, 8, padding='same', bias=False)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, 8, padding='same', bias=False)
        self.relu3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(128)

        self.glob_pool = nn.AvgPool1d()
        self.fc = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax()

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.lstm.cuda()
            self.dropout.cuda()
        logging.info(F"Model {self.name} on GPU with cuda: {self.cuda}")

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        if self.cuda:
            x = x.cuda()

        # 2D LSTM

        x_lstm = self.lstm(x)
        x_lstm = self.dropout(x_lstm)

        # 1D FCN

        x_fcn = torch.permute(x, (2, 1)).size()

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
