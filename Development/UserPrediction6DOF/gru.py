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

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim=1):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers (default 1)
        # setting batch_first=True requires the input to have the shape [batch_size, seq_len, input_size]
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Fully connected layer maps last LSTM output (hidden dimension) to the label dimension
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.lstm.cuda()
            self.fc.cuda()
        logging.info(F"Model on GPU is {self.cuda}")

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


class LSTMOptimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.cuda = torch.cuda.is_available()

    def train_step(self, x, y):
        # Sets model to train mode
        if self.cuda:
            x, y = x.cuda(), y.cuda()
        self.model.train()

        # Makes predictions
        yhat = self.model(x)
        if self.cuda:
            yhat = yhat.cuda()
        # DEBUG
        # print(f"step x: {x.shape}")
        # print(f"step y: {y.shape}")
        # print(f"step yhat: {yhat.shape}")

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=11):
        model_path = f'./models/LSTM/{datetime.now().strftime("%d.%m_%H%M%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                if self.cuda:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                # creates 3D Tensor
                x_batch = x_batch.view([batch_size, -1, n_features])
                # print(f"x_batch: {x_batch.shape}")
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    if self.cuda:
                        x_val, y_val = x_val.cuda(), y_val.cuda()
                    # creates 3D Tensor
                    x_val = x_val.view([batch_size, -1, n_features])
                    y_val = y_val
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 5) | (epoch % 5 == 0):
                # print first 5 epochs and then every 5 epochs
                logging.info(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        # saves model after training
        # torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        """
        predictions: list[float] The values predicted by the model
        values: list[float] The actual values in the test set.

        Typically validation loss should be similar to but slightly higher than training loss.
         As long as validation loss is lower than or even equal to training loss one should keep doing more training.
        If training loss is reducing without increase in validation loss then again keep doing more training
        If validation loss starts increasing then it is time to stop
        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                if self.cuda:
                    x_test, y_test = x_test.cuda(), y_test.cuda()

                x_test = x_test.view([batch_size, -1, n_features])
                y_test = y_test
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()
