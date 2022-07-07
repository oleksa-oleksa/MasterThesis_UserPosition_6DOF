import torch
import numpy as np
import logging
import time
from .pytorchtools import EarlyStopping


class NNTrainer:
    def __init__(self, model, criterion, optimizer, params):
        # TODO All parameters correct
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.cuda = torch.cuda.is_available()
        self.params = params

    def train(self, train_loader, val_loader, n_epochs=150):
        # trained model can be saved
        # model_path = f'./models/{self.model.name}_{datetime.now().strftime("%d.%m_%H%M%S")}'

        start = time.time()
        logging.info(f'{self.model.name} training started!')

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.params['patience'], verbose=False)

        for epoch in range(1, n_epochs + 1):
            batch_losses = []

            self.model.train()  # prep model for training

            for x_train_batch, y_train_batch in train_loader:
                if self.cuda:
                    x_train_batch, y_train_batch = x_train_batch.cuda(), y_train_batch.cuda()

                # print(f'x_train_batch: {x_train_batch.shape}')
                outputs_train_batch = self.model.forward(x_train_batch)  # forward pass
                # print(f'outputs_train_batch: {outputs_train_batch.shape}, y_train_batch: {y_train_batch.shape}')
                self.optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

                loss = self.criterion(outputs_train_batch, y_train_batch)
                loss.backward()  # calculates the loss of the loss function
                batch_losses.append(loss)

                self.optimizer.step()  # improve from loss, i.e backprop

            bl = [loss.detach().numpy() for loss in batch_losses]
            training_loss = np.mean(bl)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                '''
                with torch.no_grad()
                The validation loop looks very similar to training but is somewhat simplified. 
                The key difference is that validation is read-only. 
                Specifically, the loss value returned is not used, and the weights are not updated.
                '''
                batch_val_losses = []

                for x_val_batch, y_val_batch in val_loader:
                    if self.cuda:
                        x_val_batch, y_val_batch = x_val_batch.cuda(), y_val_batch.cuda()

                    self.model.eval()

                    y_val_hat = self.model(x_val_batch)
                    val_loss = self.criterion(y_val_batch, y_val_hat)
                    batch_val_losses.append(val_loss)
                vl = [loss.detach().numpy() for loss in batch_val_losses]
                validation_loss = np.mean(vl)
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

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(validation_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load('checkpoint.pt'))

        end = time.time()
        logging.info(f'\nTRAINING took {end - start}s')

        # saves model after training
        # torch.save(self.model.state_dict(), model_path)

    def predict(self, test_loader, batch_size):
        test_loss = 0.0
        self.model.eval()  # prep model for evaluation

        for data, target in test_loader:
            if len(target.data) != batch_size:
                break
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(data)
            # calculate the loss
            loss = self.criterion(output, target)
            # update test loss
            test_loss += loss.item() * data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # calculate and print avg test loss
        test_loss = test_loss / len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))