import torch
import numpy as np
import logging
import time
from UserPrediction6DOF.tools import EarlyStopping


class NNTrainer:
    def __init__(self, model, criterion, optimizer, params):
        # TODO All parameters correct
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.cuda = torch.cuda.is_available()
        self.params = params

    def train(self, train_loader, val_loader, n_epochs=150):
        # trained model can be saved
        # model_path = f'./trained_models/{self.model.name}_{datetime.now().strftime("%d.%m_%H%M%S")}'

        start = time.time()
        logging.info(f'{self.model.name} training started!')

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.params['patience'], verbose=False,
                                       delta=self.params['delta'], trace_func=logging.info)

        for epoch in range(1, n_epochs + 1):
            batch_losses = []

            self.model.train()  # prep model for training

            for x_train_batch, y_train_batch in train_loader:
                if self.cuda:
                    x_train_batch, y_train_batch = x_train_batch.cuda(), y_train_batch.cuda()

                # print(f'x_train_batch: {x_train_batch.shape}')
                y_train_batch = torch.tensor(np.round(y_train_batch.cpu().detach().numpy(), 8))
                if self.cuda:
                    y_train_batch = y_train_batch.cuda()

                outputs_train_batch = self.model.forward(x_train_batch)  # forward pass

                # logging.info(f'y_train_batch: {y_train_batch.type()}, output: {outputs_train_batch.type()}')
                # logging.info(f'outputs_train_batch: {outputs_train_batch.shape}, y_train_batch: {y_train_batch.shape}')
                self.optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

                loss = self.criterion(outputs_train_batch, y_train_batch)
                loss.backward()  # calculates the loss of the loss function
                batch_losses.append(loss)

                self.optimizer.step()  # improve from loss, i.e backprop

            bl = [loss.cpu().detach().numpy() for loss in batch_losses]
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

                    y_val_batch = torch.tensor(np.round(y_val_batch.cpu().detach().numpy(), 8))
                    if self.cuda:
                        y_val_batch = y_val_batch.cuda()

                    y_val_hat = self.model(x_val_batch)
                    val_loss = self.criterion(y_val_batch, y_val_hat)
                    batch_val_losses.append(val_loss)

            vl = [loss.cpu().detach().numpy() for loss in batch_val_losses]
            validation_loss = np.mean(vl)
            self.val_losses.append(validation_loss)

            if self.params['lr_reducing']:
                if (epoch >= self.params['lr_epochs']) & (epoch % self.params['lr_epochs'] == 0):
                    for g in self.optimizer.param_groups:
                        g['lr'] = g['lr'] * self.params['lr_multiplicator']
                        print(f"Learning rate is {g['lr']}")

            if (epoch <= 5) | (epoch % 1 == 0):
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

        '''
        # load the last checkpoint with the best model
        # self.model.load_state_dict(torch.load('checkpoint.pt'))

        # saves model after training
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), model_path)
        '''
        end = time.time()
        logging.info(f'\nTRAINING took {end - start}s')

    def predict(self, test_loader, batch_size):

        self.model.eval()  # prep model for evaluation
        predictions = np.empty((0, 7), float)
        targets = np.empty((0, 7), float)

        with torch.no_grad():
            batch_test_losses = []
            for x_test_batch, y_test_batch in test_loader:
                if len(y_test_batch.data) != batch_size:
                    break
                if self.cuda:
                    x_test_batch, y_test_batch = x_test_batch.cuda(), y_test_batch.cuda()

                y_test_batch = torch.tensor(np.round(y_test_batch.cpu().detach().numpy(), 8))
                if self.cuda:
                    y_test_batch = y_test_batch.cuda()

                y_test_hat = self.model(x_test_batch)
                test_loss = self.criterion(y_test_hat, y_test_batch)
                batch_test_losses.append(test_loss)

                last_pred = y_test_hat[:, -1, :].cpu().detach().numpy()
                last_targ = y_test_batch[:, -1, :].cpu().detach().numpy()

                predictions = np.concatenate((predictions, last_pred), axis=0)
                targets = np.concatenate((targets, last_targ))

        tl = [loss.cpu().detach().numpy() for loss in batch_test_losses]
        test_loss = np.mean(tl)
        self.test_losses.append(test_loss)

        print(f'Test loss: {test_loss:.4f}')

        print(f'predictions: {predictions.shape}')

        return predictions, targets
