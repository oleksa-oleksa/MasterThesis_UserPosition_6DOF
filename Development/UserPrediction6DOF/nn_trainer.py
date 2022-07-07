import torch
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import logging
import os
import time


class NNTrainer:
    def __init__(self, model, criterion, optimizer, params):
        # TODO All parameters correct
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.test_losses = []
        self.cuda = torch.cuda.is_available()
        self.params = params

    def train(self, train_loader, test_loader, n_epochs=150):
        start = time.time()
        logging.info(f'{self.model.name} training started!')

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_train_batch, y_train_batch in train_loader:
                if self.cuda:
                    x_train_batch, y_train_batch = x_train_batch.cuda(), y_train_batch.cuda()

                # print(f'x_train_batch: {x_train_batch.shape}')
                # reshaping to [batch, timestamps, layers, features]
                # x_train_batch = torch.unsqueeze(x_train_batch, 2)
                # print(f'x_train_batch: {x_train_batch.shape}')
                outputs_train_batch = self.model.forward(x_train_batch)  # forward pass
                self.optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

                # obtain the loss function
                # print(f'outputs_train_batch: {outputs_train_batch.shape}, y_train_batch: {y_train_batch.shape}')
                loss = self.criterion(outputs_train_batch, y_train_batch)

                loss.backward()  # calculates the loss of the loss function
                batch_losses.append(loss)

                self.optimizer.step()  # improve from loss, i.e backprop

            bl = [loss.detach().numpy() for loss in batch_losses]
            training_loss = np.mean(bl)
            self.train_losses.append(training_loss)

            if self.params['lr_reducing']:
                if (epoch >= self.params['lr_epochs']) & (epoch % self.params['lr_epochs'] == 0):
                    for g in self.optimizer.param_groups:
                        g['lr'] = g['lr'] * 0.3
                        print(f"Learning rate is {g['lr']}")

            if (epoch <= 5) | (epoch % 5 == 0):
                # print first 5 epochs and then every 5 epochs
                logging.info(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t"
                )

        end = time.time()
        logging.info(f'\nTRAINING took {end - start}s')
