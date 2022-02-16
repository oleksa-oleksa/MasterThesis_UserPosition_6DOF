import math
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class LSTMModelBase(nn.Module):

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

    def __init__(self, input_dim, hidden_dim):

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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input Gate
        self.W_input1 = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.bias_input1 = nn.Parameter(torch.Tensor(hidden_dim))
        self.W_input2 = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.bias_input2 = nn.Parameter(torch.Tensor(hidden_dim))

        # Forget Gate
        self.W_forget = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.bias_forget = nn.Parameter(torch.Tensor(hidden_dim))

        # Output Gate
        self.W_output1 = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.bias_output1 = nn.Parameter(torch.Tensor(hidden_dim))
        self.W_output2 = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.bias_output2 = nn.Parameter(torch.Tensor(hidden_dim))

        self.init_weights()

        def init_weights(self):
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

        def forward(self, X, init_states=None):

            """
            assumes x.shape represents (batch_size, sequence_size, input_size)
            init_states is a tuple with the (Ht, Ct) parameters, set to zero if not introduced

            firstly Ht and Ct represent previous cell parameter Ht_1 and Ct_1
            new Ct will be created with forget gate calculations
            new Ht will be created with output gate calculations
            """
            batch_size, sequence_length, _ = X.size()
            hidden_seq = []

            if init_states is None:
                Ht, Ct = (
                    torch.zeros(batch_size, self.hidden_size).to(X.device),
                    torch.zeros(batch_size, self.hidden_size).to(X.device),
                )

            else:
                Ht, Ct = init_states

            for t in range(sequence_length):
                Xt = X[:, t, :]

                input_layer1 = torch.sigmoid(self.W_input1 @ Ht + self.W_input1 @ Xt + self.bias_input1)

                input_layer2 = torch.tanh(self.W_input2 @ Ht + self.W_input2 @ Xt + self.bias_input2)

                input_gate = input_layer1 @ input_layer2

                forget_gate = torch.sigmoid(self.W_forget @ Ht + self.W_forget @ Xt + self.bias_forget)

                #  New long-term memory is created from previous  Ct_1
                Ct = Ct * forget_gate + input_gate

                # Gate takes the current input Xt, the previous short-term memory Ht_1 (hidden state)
                # and long-term memory Ct computed in current step and ouputs the new hidden state Ht
                output_layer1 = torch.sigmoid(self.W_output1 @ Ht + self.W_output1 @ Xt + self.bias_output1)

                output_layer2 = torch.tanh(self.W_output2 * Ct + self.bias_output2)

                output_gate = output_layer1 @ output_layer2

                Ht = output_gate

                hidden_seq.append(Ht.unsqueeze(0))

            # reshape hidden_seq p/ retornar
            hidden_seq = torch.cat(hidden_seq, dim=0)
            hidden_seq = hidden_seq.transpose(0, 1).contiguous()
            return hidden_seq, (Ht, Ct)


class LSTMNetBase(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        LSTM Network uses LSTMModel as cell structure

        input dimension dont't need to match the cell state (hidden state) dimension
        As we're working with a time series (position, rotation, velocity, speed) we have a vector on input
        The length of the hidden state is the summary of the history in LSTM
        """
        super().__init__()
        self.lstm = LSTMModelBase(input_dim, hidden_dim)  # nn.LSTM(32, 32, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        hidden_seq, (Ht, Ct) = self.lstm(X)
        X = self.fc1(X)
        return X


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim=1):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers (default 1)
        # setting batch_first=True requires the input to have the shape [batch_size, seq_len, input_size]
        self.layers = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Fully connected layer maps last LSTM output (hidden dimension) to the label dimension
        self.fc = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.layers(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

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

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)
        #print(yhat.shape)

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
                x_batch = x_batch.view([batch_size, -1, n_features])
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
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
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)

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
                x_test = x_test.view([batch_size, -1, n_features])
                y_test = y_test
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.detach().numpy())
                values.append(y_test.detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def format_predictions(self, predictions, values, df_test, scaler=1):
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        df_result = pd.DataFrame(data={"value": vals, "prediction": preds})
        df_result = df_result.sort_index()
        # df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
        return df_result

