import numpy as np
import torch
import os
cuda_path = "/mnt/output"


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as a downgrade.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.verbose = verbose
        self.counter_increased = 0
        self.counter_repeated = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.output = './results/checkpoint'
        self.file = path
        self.path = None
        if torch.cuda.is_available():
            self.path = os.path.join(cuda_path, 'job_results/checkpoint', self.file)
        else:
            self.path = os.path.join(self.output, self.file)
        self.trace_func = trace_func
        self.last_loss = None
        self.patience = patience
        self.patience_repeated = int(patience * 2.2)
        self.step = 0.0001

    def __call__(self, val_loss, model):
        if self.last_loss is None:
            self.last_loss = val_loss
            # self.save_checkpoint(self.best_loss, model)
            return

        # if loss is rising
        if val_loss > self.last_loss + self.delta:
            self.counter_increased += 1
            self.trace_func(f'Loss increased: counter: {self.counter_increased}/{self.patience} \t     {val_loss:.4f} > {self.last_loss:.4f}')
            if self.counter_increased >= self.patience:
                self.early_stop = True
        # if loss does'n not improve and remains the same
        elif val_loss == self.last_loss:
            self.counter_repeated += 1
            self.trace_func(f'No change counter: {self.counter_repeated}/{self.patience_repeated}')
            if self.counter_repeated >= self.patience_repeated:
                self.early_stop = True
        elif val_loss <= self.last_loss - self.step:
            # self.save_checkpoint(val_loss, model)
            if self.counter_increased > 0 or self.counter_repeated > 0:
                self.trace_func(f'RESET counters with loss {val_loss:.4f}')
                self.counter_increased = 0
                self.counter_repeated = 0

        # saving current loss for the next epoch
        self.last_loss = val_loss

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
