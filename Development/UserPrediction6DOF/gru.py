import math
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import logging


class GRUModel(nn.Module):
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
        super(GRUModel, self).__init__()
        self.name = "GRU"

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

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
        logging.info(F"Model {self.name} on GPU with cuda: {self.cuda}")

    def forward(self, x):
        if self.cuda:
            x = x.cuda()
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        if self.cuda:
            h0 = h0.cuda()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out