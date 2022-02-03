import torch
import torch.nn as nn


class LSTM(object):

    """
    Implements a sequential network named Long Short Term Memory Network.
    It is capable of learning order dependence in sequence prediction problems and
    can handle the vanishing gradient problem faced by RNN.

    At each iteration, the LSTM uses gating mechanism and works with:
    1. the current input data,
    2. the hidden state aka the short-term memory
    3. lastly the long-term memory.

    Gates regulate the information to be kept or discarded at each time step
    before passing on the long-term and short-term information to the next cell.


    Parameters
        ---------
        param, : array_like
            Description

    Returns
        ---------
        speed, turn : tuple
            predicted speed/turn after calculations mentioned above

    """