import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import logging


class LSTMModel1(nn.Module):
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
        super(LSTMModel1, self).__init__()
        self.name = "Basic simple LSTM with Sliding Window"

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
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.lstm.cuda()
        self.fc.cuda()

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        if self.cuda:
            x = x.cuda()
            h0, c0 = h0.cuda(), c0.cuda()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backpropogate all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # This is a basic simple LSTM
        # uses only output from LSTM and passes it through the linear layer
        out = self.fc(out)
        return out


class LSTMModel2(nn.Module):
    """
    Next you are going to use 2 LSTM layers with the same hyperparameters stacked over each other
    (via hidden_size), you have defined the 2 Fully Connected layers, the ReLU layer, and some helper variables. Next,
    you are going to define the forward pass of the LSTM
    """
    def __init__(self, seq_length_input, input_dim, hidden_dim, seq_length_output, output_dim, dropout, layer_dim):
        super(LSTMModel2, self).__init__()
        self.name = "LSTM2 with 2 FC and 2 ReLU"
        self.num_layers = layer_dim  # number of layers
        self.dropout = dropout
        self.input_size = input_dim  # input size
        self.hidden_size = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs
        self.seq_length_input = seq_length_input  # sequence length
        self.seq_length_output = seq_length_output  # otput length of timeseries in the future

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=layer_dim, batch_first=True, dropout=dropout)  # lstm
        self.relu_1 = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_dim, 128)  # fully connected 1
        self.relu_2 = nn.ReLU()
        self.fc_2 = nn.Linear(128, output_dim)  # fully connected last layer

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.lstm.cuda()
        self.relu_1.cuda()
        self.fc_1.cuda()
        self.relu_2.cuda()
        self.fc_2.cuda()
        self.fc_lstm.cuda()

    def forward(self, x):
        # print(f"x: {x.shape}")
        # define the hidden state, and internal state first, initialized with zeros
        h_0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))  # hidden state
        # print(f"h_0: {h_0.shape}")
        c_0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))  # internal state
        # print(f"c_0: {c_0.shape}")

        if self.cuda:
            x = x.cuda()
            h_0, c_0 = h_0.cuda(), c_0.cuda()

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        # print(f"lstm output: {output.shape}")
        # print(f"hn before -1: {hn.shape}")
        # hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        # print(f"hn -1: {hn.shape}")
        out = self.relu_1(output)
        # print(f"relu 1: {out.shape}")
        out = self.fc_1(out)  # First Dense
        # print(f"First Dense fc_1: {out.shape}")
        out = self.relu_2(out)  # relu
        # print(f"relu 2: {out.shape}")
        out = self.fc_2(out)  # Final Output
        # print(f"Final Output fc_2: {out.shape}")
        return out


class LSTMModel3(nn.Module):
    """
    Next you are going to use 2 LSTM layers with the same hyperparameters stacked over each other
    (via hidden_size), you have defined the 2 Fully Connected layers, the ReLU layer, and some helper variables. Next,
    you are going to define the forward pass of the LSTM
    """
    def __init__(self, seq_length_input, input_dim, hidden_dim, seq_length_output, output_dim, dropout, layer_dim):
        super(LSTMModel3, self).__init__()
        self.name = "LSTM3 with 2 FC and 2 MISH"
        self.num_layers = layer_dim  # number of layers
        self.dropout = dropout
        self.input_size = input_dim  # input size
        self.hidden_size = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs
        self.seq_length_input = seq_length_input  # sequence length
        self.seq_length_output = seq_length_output  # otput length of timeseries in the future
        self.inner_size = 128

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_dim,
                            num_layers=layer_dim, batch_first=True, dropout=dropout)  # lstm
        self.mish_1 = nn.Mish()
        self.fc_1 = nn.Linear(hidden_dim, self.inner_size)  # fully connected 1
        self.mish_2 = nn.Mish()
        self.fc_2 = nn.Linear(self.inner_size, output_dim)  # fully connected last layer
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.lstm.cuda()
        self.mish_1.cuda()
        self.fc_1.cuda()
        self.mish_2.cuda()
        self.fc_2.cuda()

    def forward(self, x):
        # print(f"x: {x.shape}")
        # define the hidden state, and internal state first, initialized with zeros
        h_0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))  # hidden state
        # print(f"h_0: {h_0.shape}")
        c_0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))  # internal state
        # print(f"c_0: {c_0.shape}")

        if self.cuda:
            x = x.cuda()
            h_0, c_0 = h_0.cuda(), c_0.cuda()

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        # print(f"lstm output: {output.shape}")
        # print(f"hn before -1: {hn.shape}")
        # hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        # print(f"hn -1: {hn.shape}")
        out = self.mish_1(output)
        # print(f"mish_1: {out.shape}")
        out = self.fc_1(out)  # First Dense
        # print(f"Second Dense fc_1: {out.shape}")
        out = self.mish_2(out)  # relu
        # print(f"mish_1: {out.shape}")
        out = self.fc_2(out)  # Final Output
        # print(f"Final Output fc_2: {out.shape}")
        return out


class LSTMModel4(nn.Module):
    """
    Next you are going to use 2 LSTM layers with the same hyperparameters stacked over each other
    (via hidden_size), you have defined the 2 Fully Connected layers, the ReLU layer, and some helper variables. Next,
    you are going to define the forward pass of the LSTM
    """
    def __init__(self, seq_length_input, input_dim, hidden_dim, seq_length_output, output_dim, dropout, layer_dim):
        super(LSTMModel4, self).__init__()
        self.name = "LSTM4 with Dropout and 3 FC and 2 MISH"
        self.num_layers = layer_dim  # number of layers
        self.dropout = dropout
        self.input_size = input_dim  # input size
        self.hidden_size = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs
        self.seq_length_input = seq_length_input  # sequence length
        self.seq_length_output = seq_length_output  # otput length of timeseries in the future
        self.inner_size = 2 * hidden_dim
        self.lstm_dropout = 0.2

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        
        self.lstm_1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=layer_dim, batch_first=True, dropout=dropout)  # lstm
        self.drop_1 = nn.Dropout3d(p=self.lstm_dropout)
        self.mish_1 = nn.Mish()

        self.lstm_2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.inner_size,
                              num_layers=layer_dim, batch_first=True, dropout=dropout)
        self.mish_2 = nn.Mish()
        self.fc_2 = nn.Linear(self.inner_size, self.hidden_size)
        
        self.relu_3 = nn.ReLU()
        self.fc_3 = nn.Linear(self.hidden_size, output_dim)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.lstm_1.cuda()
        self.drop_1.cuda()
        self.mish_1.cuda()
        self.lstm_2.cuda()
        self.mish_2.cuda()
        self.fc_2.cuda()
        self.relu_3.cuda()
        self.fc_3.cuda()

    def forward(self, x):
        # define the hidden state, and internal state first, initialized with zeros
        h_0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))  # internal state

        h_0_2 = Variable(torch.zeros(self.num_layers, x.shape[0], self.inner_size))  # hidden state
        c_0_2 = Variable(torch.zeros(self.num_layers, x.shape[0], self.inner_size))  # internal state

        if self.cuda:
            x = x.cuda()
            h_0, c_0 = h_0.cuda(), c_0.cuda()
            h_0_2, c_0_2 = h_0_2.cuda(), c_0_2.cuda()

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm_1(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        out = self.drop_1(output)
        out = self.mish_1(out)
        output, (hn_2, cn_2) = self.lstm_2(out, (h_0_2, c_0_2))  # lstm with input, hidden, and internal state
        out = self.mish_2(output)
        out = self.fc_2(out)  
        out = self.relu_3(out)
        out = self.fc_3(out)
        return out
