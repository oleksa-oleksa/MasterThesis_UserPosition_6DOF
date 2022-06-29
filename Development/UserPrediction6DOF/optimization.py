import torch
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import logging
import os


class RNNOptimization:
    def __init__(self, model, loss_fn, optimizer, results, params):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.results_path = results
        self.cuda = torch.cuda.is_available()
        self.params = params

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
        #print(f"step y: {y.shape}")
        #print(y)
        #print(f"step yhat: {yhat.shape}")
        #print(yhat)

        # Computes loss
        loss = self.loss_fn(input=yhat, target=y)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=10):
        # trained model can be saved
        # model_path = f'./models/LSTM_Sliding/{datetime.now().strftime("%d.%m_%H%M%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                if self.cuda:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                # creates 3D Tensor
                # print(f"x_batch before -1: {x_batch.shape}")
                #print(f"y_batch: {y_batch.shape}")
                # x_batch = x_batch.view([batch_size, -1, n_features])
                # print(f"x_batch after: {x_batch.shape}")
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
                    # x_val = x_val.view([batch_size, -1, n_features])
                    # y_val = y_val
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if self.params['lr_reducing']:
                if (epoch >= self.params['lr_epochs']) & (epoch % self.params['lr_epochs'] == 0):
                    for g in self.optimizer.param_groups:
                        g['lr'] = g['lr'] * 0.3
                        print(f"Learning rate is {g['lr']}")

            if (epoch <= 5) | (epoch % 5 == 0):
                # print first 5 epochs and then every 5 epochs
                logging.info(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        # saves model after training
        # torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=7):
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

                # print(f"x_test before -1: {x_test.shape}")
                # x_test = x_test.view([batch_size, -1, n_features])
                # y_test = y_test
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def predict(self, test_loader):
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
            batch_test_losses = []
            batch_counter = 0

            for x_test_batch, y_test_batch in test_loader:
                batch_counter += 1
                if self.cuda:
                    x_test_batch, y_test_batch = x_test_batch.cuda(), y_test_batch.cuda()

                # print(f"x_test before -1: {x_test_batch.shape}")
                # print(f"y_batch: {y_test_batch.shape}")

                # x_test = x_test.view([batch_size, -1, n_features])
                # y_test = y_test
                self.model.eval()
                yhat = self.model(x_test_batch)
                # print(f"yhat: {yhat.shape}")

                test_loss = self.loss_fn(y_test_batch, yhat).item()
                batch_test_losses.append(test_loss)
                testing_loss = np.mean(batch_test_losses)
                self.test_losses.append(testing_loss)

                predictions.extend(yhat.cpu().detach().numpy())
                # print(f"predictions: {np.array(predictions).shape}")
                values.extend(y_test_batch.cpu().detach().numpy())
                # print(f"values: {np.array(values).shape}")

                if batch_counter % 1000 == 0:
                    # print first 5 batches and then every 5 batches
                    logging.info(
                        f"[{batch_counter}] Test loss: {testing_loss:.4f}"
                    )

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Train/Val losses: {int(self.params['LAT']*1e3)}ms "
                  f"hidden: {self.params['hidden_dim']}, batch: "
                  f"{self.params['batch_size']}, dropout: {self.params['dropout']}, "
                  f"layers: {self.params['layers']}")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        #plt.show()

        head, _ = os.path.split(self.results_path)
        out = os.path.join(head, 'losses')
        if not os.path.exists(out):
            os.makedirs(out)
        dest = os.path.join(out, f"Fig-LAT{int(self.params['LAT']*1e3)}_"
                            f"hid{self.params['hidden_dim']}_epochs{self.params['epochs']}_"
                            f"batch{self.params['batch_size']}_drop{self.params['dropout']}_"
                            f"layers{self.params['layers']}.pdf")
        plt.savefig(dest)
        logging.info(f"Saved to file {dest}")

        plt.close()
