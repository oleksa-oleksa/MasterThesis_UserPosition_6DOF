import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def train_val_test_split(X, y, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def test_train_split(X, y, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    return X_train, X_test, y_train, y_test


def add_sliding_window(X, y, seq_length, pred_step):
    X_w = []
    y_w = []

    # SLIDING WINDOW LOOKING INTO PAST TO PREDICT 20 ROWS INTO FUTURE
    for i in range(seq_length, len(X) - pred_step + 1):
        X_w.append(X[i - seq_length:i, 0:X.shape[1]])
        y_w.append(y[i:i + pred_step, 0:y.shape[1]])

    X_w, y_w = np.array(X_w), np.array(y_w)

    logging.info("------------- Creating 3D datasets and adding sliding window ------------")
    logging.info(f'X_w.shape: {X_w.shape}')
    logging.info(f'y_w.shape: {y_w.shape}')
    logging.info(f"Sliding window of {seq_length} added and 3D datasets are created!")
    logging.info("--------")
    return X_w, y_w


def load_dataset(dataset_path):
    logging.info(f"---------------- Reading 2D dataset from {dataset_path} ----------------")
    df = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f'Columns: {list(df.columns)}')
    logging.info("--------")
    return df


def prepare_X_y(df, features, seq_length, pred_step, outputs):
    X = df[features].to_numpy()
    logging.info("------------------ Creating 2D X and y datasets  -----------------------")
    logging.info(f'X.shape: {X.shape}')
    logging.info(f'Using past {seq_length} values for predict in {pred_step} in future')

    y = df[outputs].to_numpy()
    logging.info(f'y.shape: {y.shape}')
    logging.info('2D datasets X and y created')
    logging.info("--------")
    return X, y


def prepare_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_features = torch.Tensor(X_train)
    train_targets = torch.Tensor(y_train)
    test_features = torch.Tensor(X_test)
    test_targets = torch.Tensor(y_test)

    train = TensorDataset(train_features, train_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


def load_data(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
    """

    :param X_train:
    :param X_val:
    :param X_test:
    :param y_train:
    :param y_val:
    :param y_test:
    :param batch_size:
    :return:

    The drop_last=True parameter ignores the last batch
    (when the number of examples in a dataset is not divisible
    by a batch_size) while drop_last=False will make the last batch
    smaller than a batch_size
    """

    train_features = torch.Tensor(X_train)
    train_targets = torch.Tensor(y_train)
    val_features = torch.Tensor(X_val)
    val_targets = torch.Tensor(y_val)
    test_features = torch.Tensor(X_test)
    test_targets = torch.Tensor(y_test)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, test_loader_one
