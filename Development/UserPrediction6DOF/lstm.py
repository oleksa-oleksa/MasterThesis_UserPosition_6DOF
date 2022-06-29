import math
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import logging


class LSTMModelCustom(nn.Module):

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

        """
        Instantiate an LSTM layer and provides it with the necessary arguments

        Parameters
            ---------
            input_dim  : integer
                 Size of the input Xt at each iteration
                 Input shape is (batch_size, sequence_length, feature_length)
                 The weight matrix that will multiply each element of the sequence
                 must have the shape (feature_length, output_length)
            hidden_dim : integer
                 Size of the hidden state Ht and long-term memory Ct (cell state)
        Returns
            ---------
            position, rotation : tuple of arrays
                Using given latency in ms the network outputs
                the predicted future position (x, y, z)
                and rotation in quaternion (qw, qx, qy, qz)
                for the user in virtual reality that
                to be reached after ms of latency

        """
        super().__init__()
        self.name = "LSTM Custom"
        # Defining the number of layers and the nodes in each layer

        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.input_sz = input_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()

        # Fully connected layer maps last LSTM output (hidden dimension) to the label dimension
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.fc.cuda()
        logging.info(F"Model {self.name} on GPU with cuda: {self.cuda}")

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):

        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        init_states is a tuple with the (Ht, Ct) parameters, set to zero if not introduced

        firstly Ht and Ct represent previous cell parameter Ht_1 and Ct_1
        new Ct will be created with forget gate calculations
        new Ht will be created with output gate calculations
        """
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_dim).to(x.device),
                        torch.zeros(bs, self.hidden_dim).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_dim
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # return hidden_seq, (h_t, c_t)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = hidden_seq[:, -1, :]
        # print(f"out BEFORE FC {out.shape}")

        # Convert the final state to our desired output shape (batch_size, output_dim)
        # print(f"out BEFORE {out.shape}")
        out = self.fc(out)
        # print(f"out AFTER FC {out.shape}")
        out = out.view([bs, -1, self.output_dim])
        # print(f"out AFTER -1 {out.shape}")
        return out


class LSTMModel(nn.Module):
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
        super(LSTMModel, self).__init__()
        self.name = "LSTM with Sliding Window"

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        # LSTM layers (default 1)
        # setting batch_first=True requires the input to have the shape [batch_size, seq_len, input_size]
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)

        # Fully connected layer maps last LSTM output (hidden dimension) to the label dimension
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.lstm.cuda()
            self.fc.cuda()
        logging.info(F"Init model {self.name}")

    def forward(self, x):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        # Initializing hidden state for first input with zeros
        if self.cuda:
            x = x.cuda()
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # print(f"x input in forward is {x.shape}")
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        if self.cuda:
            h0, c0 = h0.cuda(), c0.cuda()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # print(f"out after lstm before -1 {out.shape}")

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        # print(f"out after -1 BEFORE FC {out.shape}")

        # Convert the final state to our desired output shape (batch_size, output_dim)
        # print(f"out BEFORE {out.shape}")
        out = self.fc(out)
        # print(f"out AFTER FC {out.shape}")
        out = out.view([batch_size, -1, self.output_dim])
        # print(f"out AFTER -1 {out.shape}")
        return out


class LSTMModelSVP_1to1(nn.Module):
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

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, layer_dim=1, batch_size=2048):
        """Works both on CPU and GPU without additional modifications"""
        super(LSTMModel, self).__init__()
        self.name = "LSTM"

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers (default 1)
        # setting batch_first=True requires the input to have the shape [batch_size, seq_len, input_size]
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)

        # Fully connected layer maps last LSTM output (hidden dimension) to the label dimension
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.lstm.cuda()
            self.fc.cuda()
        logging.info(F"Model {self.name} on GPU with cuda: {self.cuda}")

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        if self.cuda:
            x = x.cuda()
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        if self.cuda:
            h0, c0 = h0.cuda(), c0.cuda()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        # print(f"out BEFORE FC {out.shape}")

        # Convert the final state to our desired output shape (batch_size, output_dim)
        # print(f"out BEFORE {out.shape}")
        out = self.fc(out)
        # print(f"out AFTER FC {out.shape}")

        return out
