import torch
import torch.nn as nn
import logging
import math


class GRUModel1(nn.Module):
    """
    Gated Recurrent Units (GRU) is a slightly more streamlined variant
    that provides comparable performance and considerably faster computation.
    Like LSTMs, they also capture long-term dependencies, but they do so
    by using reset and update gates without any cell state.

    While the update gate determines how much of the past information needs to be kept,
    the reset gate decides how much of the past information to forget.
    Doing fewer tensor operations, GRUs are often faster and require less memory than LSTMs.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob, layer_dim=1):
        super(GRUModel1, self).__init__()
        self.name = "Basic GRU with Sliding Window"

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.gru.cuda()
            self.fc.cuda()
        logging.info(F"Model {self.name}. GPU with cuda: {self.cuda}")

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        if self.cuda:
            x = x.cuda()
            h0 = h0.cuda()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())
        # print(f"out BEFORE FC {out.shape}")

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        # print(f"out AFTER FC {out.shape}")
        return out


class GRUModel3(nn.Module):
    """
    Next you are going to use 2 LSTM layers with the same hyperparameters stacked over each other
    (via hidden_size), you have defined the 2 Fully Connected layers, the ReLU layer, and some helper variables. Next,
    you are going to define the forward pass of the LSTM
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob, layer_dim=1):
        super(GRUModel3, self).__init__()
        self.name = "GRU3 with 2 FC and 2 MISH"
        self.layer_dim = layer_dim  # number of layers
        self.dropout = dropout_prob
        self.input_dim = input_dim  # input size
        self.hidden_dim = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs
        self.inner_size = 2 * hidden_dim

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        self.gru_1 = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.mish_1 = nn.Mish()
        self.fc_1 = nn.Linear(hidden_dim, self.inner_size)  # fully connected 1
        self.mish_2 = nn.Mish()
        self.fc_2 = nn.Linear(self.inner_size, output_dim)  # fully connected last layer
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.gru_1.cuda()
        self.mish_1.cuda()
        self.fc_1.cuda()
        self.mish_2.cuda()
        self.fc_2.cuda()

    def forward(self, x):
        # print(f"x: {x.shape}")
        # define the hidden state, and internal state first, initialized with zeros
        h_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()  # hidden state

        if self.cuda:
            x = x.cuda()
            h_0 = h_0.cuda()

        # Propagate input through GRU
        out, _ = self.gru_1(x, h_0.detach())
        # print(f"lstm output: {output.shape}")
        # print(f"hn before -1: {hn.shape}")
        # hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        # print(f"hn -1: {hn.shape}")
        out = self.mish_1(out)
        # print(f"mish_1: {out.shape}")
        out = self.fc_1(out)  # First Dense
        # print(f"Second Dense fc_1: {out.shape}")
        out = self.mish_2(out)  # relu
        # print(f"mish_1: {out.shape}")
        out = self.fc_2(out)  # Final Output
        # print(f"Final Output fc_2: {out.shape}")
        return out


class GRUModel31(nn.Module):
    """
    Next you are going to use 2 LSTM layers with the same hyperparameters stacked over each other
    (via hidden_size), you have defined the 2 Fully Connected layers, the ReLU layer, and some helper variables. Next,
    you are going to define the forward pass of the LSTM
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel31, self).__init__()
        self.name = "GRU3 with 2 FC and 2 MISH"
        self.layer_dim = 1  # number of layers
        self.dropout = 0
        self.input_dim = input_dim  # input size
        self.hidden_dim = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs
        self.inner_size = 2 * hidden_dim

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        self.gru_1 = nn.GRU(input_dim, hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout)
        self.mish_1 = nn.Mish()
        self.fc_1 = nn.Linear(hidden_dim, self.output_dim)  # fully connected 1
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.gru_1.cuda()
        self.mish_1.cuda()
        self.fc_1.cuda()

    def forward(self, x):
        # print(f"x: {x.shape}")
        # define the hidden state, and internal state first, initialized with zeros
        h_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()  # hidden state

        if self.cuda:
            x = x.cuda()
            h_0 = h_0.cuda()

        # Propagate input through GRU
        out, _ = self.gru_1(x, h_0.detach())
        out = self.mish_1(out)
        out = self.fc_1(out)
        return out


class GRUModel32(nn.Module):
    """
    Next you are going to use 2 LSTM layers with the same hyperparameters stacked over each other
    (via hidden_size), you have defined the 2 Fully Connected layers, the ReLU layer, and some helper variables. Next,
    you are going to define the forward pass of the LSTM
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel32, self).__init__()
        self.name = "Stacked GRU33 with MISH"
        self.layer_dim = 1  # number of layers
        self.dropout = 0
        self.input_dim = input_dim  # input size
        self.hidden_dim = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        self.gru_1 = nn.GRU(input_dim, hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout)
        self.mish_1 = nn.Mish()
        self.gru_2 = nn.GRU(hidden_dim, hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout)
        self.mish_2 = nn.Mish()
        self.gru_3 = nn.GRU(hidden_dim, hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout)
        self.mish_3 = nn.Mish()
        self.fc_3 = nn.Linear(hidden_dim, output_dim)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.gru_1.cuda()
        self.mish_1.cuda()
        self.gru_2.cuda()
        self.mish_2.cuda()
        self.gru_3.cuda()
        self.mish_3.cuda()
        self.fc_3.cuda()

    def forward(self, x):
        # print(f"x: {x.shape}")
        # define the hidden state, and internal state first, initialized with zeros
        h_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()  # hidden state
        h_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()  # hidden state
        h_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()  # hidden state

        if self.cuda:
            x = x.cuda()
            h_1 = h_1.cuda()
            h_2 = h_2.cuda()
            h_3 = h_3.cuda()

        # Propagate input through GRU with Mish Activation Layers
        out, h_2 = self.gru_1(x, h_1.detach())
        out = self.mish_1(out)
        out, h_3 = self.gru_2(out, h_2.detach())
        out = self.mish_2(out)
        out, _ = self.gru_3(out, h_3.detach())
        out = self.mish_3(out)
        out = self.fc_3(out)
        return out


class GRUModel33(nn.Module):
    """
    Next you are going to use 2 LSTM layers with the same hyperparameters stacked over each other
    (via hidden_size), you have defined the 2 Fully Connected layers, the ReLU layer, and some helper variables. Next,
    you are going to define the forward pass of the LSTM
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel33, self).__init__()
        self.name = "Stacked GRU33 with MISH and Dropout"
        self.layer_dim = 1  # number of layers
        self.dropout = 0.3
        self.input_dim = input_dim  # input size
        self.hidden_dim = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs
        self.inner_size = math.floor(hidden_dim / 2)

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        self.gru_1 = nn.GRU(input_dim, self.inner_size, self.layer_dim, batch_first=True, dropout=self.dropout)
        self.mish_1 = nn.Mish()
        self.drop_1 = nn.Dropout3d(p=self.dropout)

        self.gru_2 = nn.GRU(self.inner_size, hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout)
        self.mish_2 = nn.Mish()
        self.drop_2 = nn.Dropout3d(p=self.dropout)

        self.gru_3 = nn.GRU(hidden_dim, hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout)
        self.mish_3 = nn.Mish()

        self.pool = nn.AdaptiveMaxPool1d(output_size=output_dim)

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.gru_1.cuda()
        self.mish_1.cuda()
        self.drop_1.cuda()
        self.gru_2.cuda()
        self.mish_2.cuda()
        self.drop_2.cuda()
        self.gru_3.cuda()
        self.mish_3.cuda()
        self.pool.cuda()

    def forward(self, x):
        # print(f"x: {x.shape}")
        # define the hidden state, and internal state first, initialized with zeros
        h_1 = torch.zeros(self.layer_dim, x.size(0), self.inner_size).requires_grad_()  # hidden state
        h_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()  # hidden state
        h_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()  # hidden state

        if self.cuda:
            x = x.cuda()
            h_1 = h_1.cuda()
            h_2 = h_2.cuda()
            h_3 = h_3.cuda()

        # Propagate input through GRU with Mish Activation Layers
        out, h_2 = self.gru_1(x, h_1.detach())
        out = self.mish_1(out)
        out = self.drop_1(out)

        out, h_3 = self.gru_2(out, h_2.detach())
        out = self.mish_2(out)
        out = self.drop_2(out)

        out, _ = self.gru_3(out, h_3.detach())
        out = self.mish_3(out)
        out = self.pool(out)

        return out


class GRUModel34(nn.Module):
    """
    Next you are going to use 2 LSTM layers with the same hyperparameters stacked over each other
    (via hidden_size), you have defined the 2 Fully Connected layers, the ReLU layer, and some helper variables. Next,
    you are going to define the forward pass of the LSTM
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel34, self).__init__()
        self.name = "GRU34 using GRU1 with MISH and Dropout"
        self.layer_dim = 1  # number of layers
        self.dropout = 0.3
        self.input_dim = input_dim  # input size
        self.hidden_dim = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs
        self.inner_size = math.floor(hidden_dim / 2)

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        self.gru_1 = nn.GRU(input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=0)
        self.mish_1 = nn.Mish()
        self.drop_1 = nn.Dropout3d(p=self.dropout)
        self.fc_1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.gru_1.cuda()
        self.mish_1.cuda()
        self.drop_1.cuda()
        self.fc_1.cuda()

    def forward(self, x):
        h_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()  # hidden state

        if self.cuda:
            x = x.cuda()
            h_1 = h_1.cuda()

        # Propagate input through GRU with Mish Activation Layers
        out, _ = self.gru_1(x, h_1.detach())
        out = self.mish_1(out)
        out = self.drop_1(out)
        out = self.fc_1(out)
        return out


class GRUModel35(nn.Module):
    """
    Next you are going to use 2 LSTM layers with the same hyperparameters stacked over each other
    (via hidden_size), you have defined the 2 Fully Connected layers, the ReLU layer, and some helper variables. Next,
    you are going to define the forward pass of the LSTM
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel35, self).__init__()
        self.name = "GRU34 using GRU1 with MISH and Dropout"
        self.layer_dim = 1  # number of layers
        self.dropout = 0.05
        self.input_dim = input_dim  # input size
        self.hidden_dim = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs
        self.inner_size = math.floor(hidden_dim / 2)

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        self.gru_1 = nn.GRU(input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=0)
        # self.mish_1 = nn.Mish()
        self.drop_1 = nn.Dropout3d(p=self.dropout)
        self.fc_1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.convert_to_cuda()
        logging.info(F"Init model {self.name}")

    def convert_to_cuda(self):
        self.gru_1.cuda()
        # self.mish_1.cuda()
        self.drop_1.cuda()
        self.fc_1.cuda()


    def forward(self, x):
        h_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()  # hidden state

        if self.cuda:
            x = x.cuda()
            h_1 = h_1.cuda()

        # Propagate input through GRU with Mish Activation Layers
        out, _ = self.gru_1(x, h_1.detach())
        # out = self.mish_1(out)
        out = self.drop_1(out)
        out = self.fc_1(out)
        return out


