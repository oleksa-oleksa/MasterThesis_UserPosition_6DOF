import numpy as np
import pandas as pd
import sys


def find_min():
    df = pd.read_csv(sys.argv[2], skipfooter=1, engine='python')
    print(df)
    minMSE_pos = df['MSE_pos'].min()
    print(f"MIN MSE_pos: {minMSE_pos}")
    parameters = df.loc[df['MSE_pos'] == minMSE_pos]
    hs = parameters['hidden_dim'].values[0]
    all_hs = df.loc[df['hidden_dim'] == hs].sort_values(by=['MSE_pos']).head(10)
    print(all_hs)


def get_LSTM_train_size(batch_size, test_percent):
    df = pd.read_csv(sys.argv[2], skipfooter=1, engine='python')
    # substract test_percent to be excluded from training, reserved for testset
    number_of_samples = df.shape[0]
    print("# Shape of the input dataframe",number_of_samples)
    number_of_samples *= 1 - test_percent
    train_set_sizes = []
    for size in range(int(number_of_samples) - (batch_size - 100), int(number_of_samples)):
        mod=size%batch_size
        if (mod == 0):
            train_set_sizes.append(size)
            print(size)
    max_ts = (max(train_set_sizes))
    print(f"For {batch_size} and {test_percent} max trainset is: {max_ts}")
    return (max_ts)


if __name__ == "__main__":

    print('Usage: process_results command file_path parameters')

    if sys.argv[1] == "min":
        find_min()
    elif sys.argv[1] == "train_size":
        get_LSTM_train_size(int(sys.argv[3]), float(sys.argv[4]))
