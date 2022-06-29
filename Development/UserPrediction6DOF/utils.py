# '''
# The copyright in this software is being made available under this Software
# Copyright License. This software may be subject to other third party and
# contributor rights, including patent rights, and no such rights are
# granted under this license.
# Copyright (c) 1995 - 2021 Fraunhofer-Gesellschaft zur Förderung der
# angewandten Forschung e.V. (Fraunhofer)
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted for purpose of testing the functionalities of
# this software provided that the following conditions are met:
# *     Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# *     Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# *     Neither the names of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
# WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
# COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
# NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.
# '''

import os
import logging
import numpy as np
import pandas as pd
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import csv
import math

pd.options.mode.chained_assignment = None
# HoloLens CSV-Log parameter
pos_size = 3
rot_size = 4
labels_stop = pos_size + rot_size
data_lenght = 120020 # 120020 gives 72000 rows for training and 24000 for test/validation


def preprocess_trace(trace_path, dt, out_dir):
    """
    Resample and interpolate a raw Hololens trace
    (which contains unevenly sampled data) with
    a given sampling frequency (e.g. 5ms) and write the output to a csv file.
    Arguments:
        trace_path: Path to the raw HoloLens trace.
        dt: Desired time distance between two samples [s]
        out_dir: Output directory containing the interpolated traces.
    Outputs:
        df_intp: Dataframe containing interpolated position (x,y,z) and rotation values (quaternions and Euler angles with equal spacing.
    """

    case = os.path.splitext(os.path.basename(trace_path))[0]
    # A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes.
    df = pd.read_csv(trace_path, skipfooter=1, engine='python')

    # convert seconds of timestamp obtained with Time.time function of raw HoloLens trace to nanoseconds
    df['timestamp'] *= 1000000000
    # Start the timestamp from 0
    df['timestamp'] -= df['timestamp'].iloc[0]
    df = df.astype(float)
    # Quaternion samples for Slerp
    qs = df.loc[:, ['timestamp', 'qx', 'qy', 'qz', 'qw']].to_numpy()

    ######################################
    # Resample and interpolate the position samples (x,y,z) onto a uniform grid
    df_t = df.loc[:, 'timestamp':'z']
    df_t['timestamp'] = pd.to_timedelta(df_t['timestamp'], unit='ns')
    # timestamp is column that is used instead of index for resampling.
    df_t_intp = df_t.resample(str(dt*1e3) + 'L', on='timestamp').mean().interpolate('linear')

    # interpolated position samples (x,y,z) without timestamp and index
    t_intp = df_t_intp.to_numpy()

    ######################################
    # Resample and interpolate the quaternion samples
    rots = R.from_quat(qs[:, 1:])
    times = qs[:, 0]
    slerp = Slerp(times, rots)  # Spherical Linear Interpolation of Rotations (SLERP)
    # interpolated float timestamp
    t = df_t_intp.index.to_numpy().astype(float)
    rots_intp = slerp(t)
    # interpolated quaternion samples (x,y,z,w) without timestamp and index
    q_intp = rots_intp.as_quat()

    ######################################
    # Resample and interpolate the velocities and speed samples (x,y,z) onto a uniform grid
    df_v = df.loc[:, ['timestamp', 'velocity_x', 'velocity_y', 'velocity_z', 'speed']]
    df_v['timestamp'] = pd.to_timedelta(df_v['timestamp'], unit='ns')
    df_v_intp = df_v.resample(str(dt*1e3) + 'L', on='timestamp').mean().interpolate('linear')

    # interpolated position samples (x,y,z) without timestamp and index
    v_intp = df_v_intp.to_numpy()

    ######################################
    # Compute Euler angles for the interpolated quaternion samples
    e_intp = rots_intp.as_euler('ZXY', degrees=True)

    # Combine the interpolated array and create a DataFrame
    intp = np.hstack((t[:, np.newaxis], t_intp, q_intp, v_intp, e_intp))
    df_intp = pd.DataFrame(intp, columns=np.hstack((df.columns, ['roll', 'pitch', 'yaw'])))

    # Save interpolated DataFrame to csv
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_intp = df_intp.iloc[:data_lenght]   # Make length of all traces the same.
    df_intp.to_csv(os.path.join(out_dir, case + '.csv'), index=False)
    return df_intp


def flip_negative_quaternions(trace_path, out_dir):
    """
    Normalizes interpolated Hololens trace
    Writes csv-files into out_dir
        trace_path: Path to the raw HoloLens trace.
        out_dir: Output directory containing the interpolated traces.
    Outputs:
        df_norm: Dataframe containing untouched position (x,y,z) and Euler angles
        as they were passed into function
        and normalized rotation values (quaternions).
    """
    case = os.path.splitext(os.path.basename(trace_path))[0]
    # A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes.
    df = pd.read_csv(trace_path, skipfooter=1, engine='python')

    df.loc[(df.qw < 0), 'qx'] *= -1
    df.loc[(df.qw < 0), 'qy'] *= -1
    df.loc[(df.qw < 0), 'qz'] *= -1
    df.loc[(df.qw < 0), 'qw'] *= -1
    print(df)

    # Save flipped DataFrame to csv
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(os.path.join(out_dir, case + '.csv'), index=False)
    return df


def normalize_dataset(trace_path, out_dir, norm_type, dataset_path):
    case = os.path.splitext(os.path.basename(trace_path))[0]
    # A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes.
    df = pd.read_csv(trace_path, skipfooter=1, engine='python')

    print(df.mean())

    if norm_type == "mean":
        df = (df - df.mean())/df.std()
    elif norm_type == 'min-max':
        df = (df - df.min()) / (df.max() - df.min())
    elif norm_type == 'min-max-double':
        df = 2 * ((df - df.min()) / (df.max() - df.min())) - 1
    print(df.mean())

    # Save flipped DataFrame to csv
    dset_type = os.path.basename(os.path.normpath(dataset_path))
    dest = os.path.join(out_dir + '_' + dset_type + '_' + norm_type)
    print(dest)
    if not os.path.exists(dest):
        os.makedirs(dest)
    df.to_csv(os.path.join(dest, case + '.csv'), index=False)
    logging.info(f"Normalized traces written to {dest}")
    return df


def load_dataset(dataset_path):
    logging.info(f"---------------- Reading 2D dataset from {dataset_path} ----------------")
    df = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f'Columns: {list(df.columns)}')
    logging.info("--------")
    return df


def prepare_X_y(df, features, num_past, pred_step, outputs):
    X = df[features].to_numpy()
    logging.info("------------------ Creating 2D X and y datasets  -----------------------")
    logging.info(f'X.shape: {X.shape}')
    logging.info(f'Using past {num_past} values for predict in {pred_step} in future')

    y = df[outputs].to_numpy()
    logging.info(f'y.shape: {y.shape}')
    logging.info('2D datasets X and y created')
    logging.info("--------")
    return X, y


def add_sliding_window(X, y, num_past, pred_step):
    X_w = []
    y_w = []

    # SLIDING WINDOW LOOKING INTO PAST TO PREDICT 20 ROWS INTO FUTURE
    for i in range(num_past, len(X) - pred_step + 1):
        X_w.append(X[i - num_past:i, 0:X.shape[1]])
        y_w.append(y[i:i + pred_step, 0:y.shape[1]])

    X_w, y_w = np.array(X_w), np.array(y_w)

    logging.info("------------- Creating 3D datasets and adding sliding window ------------")
    logging.info(f'X_w.shape: {X_w.shape}')
    logging.info(f'y_w.shape: {y_w.shape}')
    logging.info(f"Sliding window of {num_past} added and 3D datasets are created!")
    logging.info("--------")
    return X_w, y_w


def save_numpy_array(dataset_path, filename, np_array):
    np.save(os.path.join(dataset_path, f'{filename}.npy'), np_array)
    logging.info(f'WRITE: {filename} saved to {dataset_path}')


def load_numpy_array(dataset_path, filename):
    data = np.load(os.path.join(dataset_path, f'{filename}.npy'))
    logging.info(f'READ: {filename} loaded from {dataset_path}')
    logging.info(f'{filename}.shape: {data.shape}')
    return data


def get_csv_files(dataset_path):
    """
    Generator function to recursively output the CSV files in a directory and its sub-directories.
    Arguments:
        dataset_path: Path to the directory containing the CSV files.
    Outputs:
        Paths of the found CSV files.
    """
    numerical_files = []
    numerical_files_sorted = []
    for f in os.listdir(dataset_path):
        if not os.path.isdir(f):
            file_name, extension = f.split('.')
            if extension == "csv" and file_name.isnumeric():
                numerical_files.append(file_name)
            else:
                logging.warning("Invalid file: {}. Ignoring...".format(f))

    numerical_filenames_ints = [int(f) for f in numerical_files]
    numerical_filenames_ints.sort()

    for f in numerical_filenames_ints:
        file = str(f) + ".csv"
        numerical_files_sorted.append(os.path.join(dataset_path, file))
    return numerical_files_sorted


def cut_extra_labels(y):
    return y[:, :labels_stop]


def cut_dataset_lenght(X, y):
    """
    Assuming that X (features) is always bigger than created y (labels)
    Cuts the longer X array and makes X and y arrays to have a same length
    :param X: dataset contains user position, rotation, velocity and speed
    :param y: labels to be predicted: user position and rotation
    :return: features X with modified length
    """

    if X.shape[0] < y.shape[0]:
        sys.exit("X is shorter than y, can not cut or extend X. Please check the inputs and try again. Terminated!")

    if X.shape[0] == y.shape[0]:
        print("X is of the same shape as y, will not cut X. Continued...")
        return X

    else:
        X = X[:y.shape[0], :]
        # print("Length of X is equal of y: ", X.shape[0] == y.shape[0])
    return X


def train_val_test_split(X, y, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


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


def print_result(predictions, values, start_row, stop_row):
    """
    prints on terminal 10 elements of array near the end of prediction.
    we expect the model predicts better on the end of the dataset
    """
    logging.info("---------------- COMPARE ---------------------------------")
    logging.info(f"predictions.shape: {predictions.shape}")
    logging.info(predictions[start_row:stop_row, :])

    logging.info("------------------- REAL VALUES -----------------------------------")
    logging.info(f"values.shape: {values.shape}")
    logging.info(values[start_row:stop_row, :])

    logging.info("-------------------------------------------------------------")


def log_parameters(df_results, params):
    result_path = ""
    if torch.cuda.is_available():
        result_path = "/mnt/output/job_results"
    if not torch.cuda.is_available():
        result_path = os.path.join(os.getcwd(), 'results')
    csv_file = "model_parameters_adjust_log.csv"
    log_path = os.path.join(result_path, csv_file)
    csv_columns = ['MAE_pos', 'MAE_rot', 'RMSE_pos', 'RMSE_rot', 'LAT', 'hidden_dim',
                   'epochs', 'batch_size', 'dropout', 'layers', 'model', 'num_past',
                   'lr', 'lr_reducing', 'weight_decay', 'lr_epochs']
    file_exists = os.path.isfile(log_path)

    # model evaluation results
    dict_data = [
        {'MAE_pos': df_results.iloc[0]["mae_euc"], 'MAE_rot': df_results.iloc[0]["mae_ang"],
         'RMSE_pos': df_results.iloc[0]["rmse_euc"], 'RMSE_rot': df_results.iloc[0]["rmse_ang"]}]

    # adding model parameters
    dict_data[0].update(params)

    try:
        with open(log_path, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            if not file_exists:
                writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    logging.info(f"log_path file {log_path} exists: {os.path.exists(log_path)}")
    logging.info(f"Saved model parameters to file: {csv_file}")