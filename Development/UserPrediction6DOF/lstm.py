import math
import torch
import torch.nn as nn


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

    The output on each interation can be accessed through hidden state Ht

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