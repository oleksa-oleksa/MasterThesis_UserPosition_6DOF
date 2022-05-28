import numpy as np
import pandas as pd
import sys


def find_min():
    df = pd.read_csv(sys.argv[2], skipfooter=1, engine='python')
    mins_pos = df.nsmallest(10, 'MAE_pos')
    print(f"MAE Position MIN is {df['MAE_pos'].min()}")
    print(mins_pos)
    mins_rot = df.nsmallest(10, 'MAE_rot')
    print(f"MAE Rotation MIN is {df['MAE_rot'].min()}")
    print(mins_rot)


def find_max():
    df = pd.read_csv(sys.argv[2], skipfooter=1, engine='python')
    maxs_pos = df.nlargest(10, 'MAE_pos')
    print("MAE Position MAX")
    print(maxs_pos)
    maxs_rot = df.nlargest(10, 'MAE_rot')
    print("MAE Rotation MAX")
    print(maxs_rot)


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
    if sys.argv[1] == "max":
        find_max()
    elif sys.argv[1] == "train_size":
        get_LSTM_train_size(int(sys.argv[3]), float(sys.argv[4]))
