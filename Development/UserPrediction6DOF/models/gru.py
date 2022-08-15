import torch
import torch.nn as nn
import logging


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
        if self.cuda:
            x = x.cuda()
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        if self.cuda:
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
    def __init__(self, seq_length_input, input_dim, hidden_dim, seq_length_output, output_dim, dropout, layer_dim):
        super(GRUModel3, self).__init__()
        self.name = "LSTM3 with 2 FC and 2 MISH"
        self.num_layers = layer_dim  # number of layers
        self.dropout = dropout
        self.input_size = input_dim  # input size
        self.hidden_size = hidden_dim  # hidden state
        self.output_dim = output_dim  # outputs
        self.seq_length_input = seq_length_input  # sequence length
        self.seq_length_output = seq_length_output  # otput length of timeseries in the future
        self.inner_size = 2 * hidden_dim

        # with batch_first = True, only the input and output tensors are reported with batch first.
        # The initial memory states (h_init and c_init) are still reported with batch second.
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)
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