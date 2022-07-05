import torch
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import logging
import os


class NNTrainer:
    def __init__(self, model, loss_fn, optimizer, results, params):
        # TODO All parameters correct 
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.results_path = results
        self.cuda = torch.cuda.is_available()
        self.params = params