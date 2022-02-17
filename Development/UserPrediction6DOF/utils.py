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

import pickle
import os
import logging
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

pd.options.mode.chained_assignment = None
# HoloLens CSV-Log parameter
pos_size = 3
rot_size = 4
labels_stop = pos_size + rot_size


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
    df_intp = df_intp.iloc[:12001]   # Make length of all traces the same.
    df_intp.to_csv(os.path.join(out_dir, case + '.csv'), index=False)
    
    return df_intp


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


def print_result(predictions, values):
    """
    prints on terminal 10 elements of array near the end of prediction.
    we expect the model predicts better on the end of the dataset
    """
    print("---------------- PREDICTIONS ---------------------------------")
    print(f"predictions.shape: {predictions.shape}")
    print(predictions[2000:2010, :])

    print("------------------- VALUES -----------------------------------")
    print(f"values.shape: {values.shape}")
    print(values[2000:2010, :])
    
    print("-------------------------------------------------------------")
