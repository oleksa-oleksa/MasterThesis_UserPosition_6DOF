# '''
# The copyright in this software is being made available under this Software
# Copyright License. This software may be subject to other third party and
# contributor rights, including patent rights, and no such rights are
# granted under this license.
# Copyright (c) 1995 - 2022 Fraunhofer-Gesellschaft zur Förderung der
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

import json
import os
import pandas as pd
import numpy as np
import toml
import logging
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from UserPrediction6DOF.tools import utils
import statsmodels.api as sm
import seaborn as sns

style_path = os.path.join(os.getcwd(), 'UserPrediction6DOF/style.json')
style = json.load(open(style_path))
config_path = os.path.join(os.getcwd(), 'config.toml')
cfg = toml.load(config_path)
dataset_lengh_sec = 600


class DataPlotter():
    """Plots dataset traces"""
    @staticmethod
    def plot_datasets(dataset_path, output_path):
        logging.info(f"Plotting from {dataset_path} and saving to {output_path}")
        for trace_path in utils.get_csv_files(dataset_path):
            df = pd.read_csv(trace_path)
            ts = np.arange(0, dataset_lengh_sec + cfg['dt'], cfg['dt'])

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

            # Plot position
            ax1.plot(ts, df.loc[:len(ts) - 1, 'x'], label='x')
            ax1.plot(ts, df.loc[:len(ts) - 1, 'y'], label='y', linestyle='--')
            ax1.plot(ts, df.loc[:len(ts) - 1, 'z'], label='z', linestyle='-.')
            ax1.set_ylabel('meters')
            ax1.set_xlim(0, dataset_lengh_sec)
            ax1.legend(loc='upper left')
            ax1.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
            ax1.xaxis.set_major_locator(MultipleLocator(10))

            # Plot orientation in Quaternions
            '''
            Q =  [qx, qy, qz, qw] = qv + qw, where 
            qw is the real part  and 
            qv = iqx + jqy + kqz= (qx, qy, qz) 
            is the imaginary part 
            x, y and z represent a vector. w is a scalar that stores the rotation around the vector.
            '''
            ax2.plot(ts, df.loc[:len(ts) - 1, 'qx'], label='qx', linestyle='solid')
            ax2.plot(ts, df.loc[:len(ts) - 1, 'qy'], label='qy', linestyle='--')
            ax2.plot(ts, df.loc[:len(ts) - 1, 'qz'], label='qz', linestyle='-.')
            ax2.plot(ts, df.loc[:len(ts) - 1, 'qw'], label='real qw', linestyle='solid')
            ax2.set_xlabel('seconds')
            ax2.set_ylabel('degrees')
            ax2.set_xlim(0, dataset_lengh_sec)
            ax2.legend(loc='upper left')
            ax2.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
            ax2.xaxis.set_major_locator(MultipleLocator(10))

            # Plot orientation in Euler angles
            ax3.plot(ts, df.loc[:len(ts) - 1, 'yaw'], label='yaw')
            ax3.plot(ts, df.loc[:len(ts) - 1, 'pitch'], label='pitch', linestyle='--')
            ax3.plot(ts, df.loc[:len(ts) - 1, 'roll'], label='roll', linestyle='-.')
            ax3.set_xlabel('seconds')
            ax3.set_ylabel('degrees')
            ax3.set_xlim(0, dataset_lengh_sec)
            ax3.legend(loc='upper left')
            ax3.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
            ax3.xaxis.set_major_locator(MultipleLocator(10))

            dset_type = os.path.basename(os.path.normpath(dataset_path))
            trace_id = os.path.splitext(os.path.basename(trace_path))[0]
            dest = os.path.join(output_path, f"Fig-{trace_id}-{dset_type}.pdf")
            fig.savefig(dest)
            logging.info("Plotting trace {} and saving to file {}".format(trace_path, dest))

    """Plots flipped quaternions dataset traces"""
    @staticmethod
    def plot_datasets_quaternions_flipped(dataset_path, output_path, dataset_type):
        logging.info(f"Plotting from {dataset_path} and saving to {output_path}")
        for trace_path in utils.get_csv_files(dataset_path):
            df = pd.read_csv(trace_path)
            ts = np.arange(0, dataset_lengh_sec + cfg['dt'], cfg['dt'])

            fig, (ax) = plt.subplots(1, 1, figsize=(18, 4), sharex=True)


            # Plot orientation in Quaternions
            '''
            Q =  [qx, qy, qz, qw] = qv + qw, where 
            qw is the real part  and 
            qv = iqx + jqy + kqz= (qx, qy, qz) 
            is the imaginary part 
            x, y and z represent a vector. w is a scalar that stores the rotation around the vector.
            '''
            ax.plot(ts, df.loc[:len(ts) - 1, 'qx'], label='qx', linestyle='solid')
            ax.plot(ts, df.loc[:len(ts) - 1, 'qy'], label='qy', linestyle='--')
            ax.plot(ts, df.loc[:len(ts) - 1, 'qz'], label='qz', linestyle='-.')
            ax.plot(ts, df.loc[:len(ts) - 1, 'qw'], label='real qw', linestyle='solid')
            ax.set_xlabel('seconds')
            ax.set_ylabel('degrees')
            ax.set_xlim(0, dataset_lengh_sec)
            plt.ylim(-1.0, 1.0)
            ax.legend(loc='upper left')
            ax.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
            ax.xaxis.set_major_locator(MultipleLocator(10))

            trace_id = os.path.splitext(os.path.basename(trace_path))[0]
            dest = os.path.join(output_path, f"Fig-{trace_id}-{dataset_type}.pdf")
            fig.savefig(dest)
            logging.info("Plotting trace {} and saving to file {}".format(trace_path, dest))

    """Plots flipped quaternions dataset traces"""
    @staticmethod
    def plot_comparison(dataset_path1, dataset_path2, output_path):
        # start and end of plot. Numbers obtained empirically from plots,
        # the most significant
        start = 162
        end = 182
        ts_start = int(start/0.005)
        ts_end = int(end/0.005)
        logging.info(f"Plotting from {dataset_path1} and from {dataset_path2} and saving to {output_path}")
        for trace_path1 in utils.get_csv_files(dataset_path1):
            for trace_path2 in utils.get_csv_files(dataset_path2):
                # plotting only interpolated and flipped quaternions of the same datasets
                if (os.path.splitext(os.path.basename(trace_path1))[0]) == \
                        (os.path.splitext(os.path.basename(trace_path2))[0]):
                    df1 = pd.read_csv(trace_path1)
                    df2 = pd.read_csv(trace_path2)
                    ts = np.arange(0, dataset_lengh_sec + cfg['dt'], cfg['dt'])
                    ts_comp = ts[ts_start:ts_end]

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

                    # Plot orientation in Quaternions
                    '''
                    Q =  [qx, qy, qz, qw] = qv + qw, where 
                    qw is the real part  and 
                    qv = iqx + jqy + kqz= (qx, qy, qz) 
                    is the imaginary part 
                    x, y and z represent a vector. w is a scalar that stores the rotation around the vector.
                    '''
                    ax1.plot(ts_comp, df1.loc[ts_start:ts_end-1, 'qx'], label='qx', linestyle='solid')
                    ax1.plot(ts_comp, df1.loc[ts_start:ts_end-1, 'qy'], label='qy', linestyle='--')
                    ax1.plot(ts_comp, df1.loc[ts_start:ts_end-1, 'qz'], label='qz', linestyle='-.')
                    ax1.plot(ts_comp, df1.loc[ts_start:ts_end-1, 'qw'], label='real qw', linestyle='solid')
                    ax1.set_xlabel('seconds')
                    ax1.set_ylabel('degrees')
                    ax1.set_xlim(start, end)
                    plt.ylim(-1.0, 1.0)
                    ax1.legend(loc='upper left')
                    ax1.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
                    ax1.xaxis.set_major_locator(MultipleLocator(10))

                    ax2.plot(ts_comp, df2.loc[ts_start:ts_end-1, 'qx'], label='qx', linestyle='solid')
                    ax2.plot(ts_comp, df2.loc[ts_start:ts_end-1, 'qy'], label='qy', linestyle='--')
                    ax2.plot(ts_comp, df2.loc[ts_start:ts_end-1, 'qz'], label='qz', linestyle='-.')
                    ax2.plot(ts_comp, df2.loc[ts_start:ts_end-1, 'qw'], label='real qw', linestyle='solid')
                    ax2.set_xlabel('seconds')
                    ax2.set_ylabel('degrees')
                    ax2.set_xlim(start, end)
                    ax1.set_ylim(-1.0, 1.0)
                    ax2.legend(loc='upper left')
                    ax2.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
                    ax2.xaxis.set_major_locator(MultipleLocator(10))

                    trace_id = os.path.splitext(os.path.basename(trace_path1))[0]
                    dest = os.path.join(output_path, f"Fig-{trace_id}-compare.pdf")
                    fig.savefig(dest)
                    logging.info("Plotting trace {} and saving to file {}".format(trace_path1, dest))

    """Plots flipped quaternions dataset traces"""

    @staticmethod
    def plot_position(dataset_path, output_path):
        logging.info(f"Plotting from {dataset_path} and saving to {output_path}")
        print(len([name for name in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, name))]))

        for trace_path in utils.get_csv_files(dataset_path):
            df = pd.read_csv(trace_path)
            ts = np.arange(0, dataset_lengh_sec + cfg['dt'], cfg['dt'])

            fig, (ax) = plt.subplots(1, 1, figsize=(18, 4), sharex=True)

            # Plot position in Quaternions
            ax.plot(ts, df.loc[:len(ts) - 1, 'x'], label='qx', linestyle='solid')
            ax.plot(ts, df.loc[:len(ts) - 1, 'y'], label='qy', linestyle='--')
            ax.plot(ts, df.loc[:len(ts) - 1, 'z'], label='qz', linestyle='-.')
            ax.set_xlabel('seconds')
            ax.set_ylabel('degrees')
            ax.set_xlim(0, dataset_lengh_sec)
            plt.ylim(-15.0, 15.0)
            ax.legend(loc='upper left')
            ax.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
            ax.xaxis.set_major_locator(MultipleLocator(10))

            trace_id = os.path.splitext(os.path.basename(trace_path))[0]
            dest = os.path.join(output_path, f"Fig-{trace_id}-position.pdf")
            fig.savefig(dest)
            logging.info("Plotting trace {} and saving to file {}".format(trace_path, dest))

    @staticmethod
    def plot_autocorrelation(dataset_path, output_path, dataset_type):
        for trace_path in utils.get_csv_files(dataset_path):
            df = pd.read_csv(trace_path, skipfooter=1, engine='python')
            plt.rc("figure", figsize=(10, 6))
            sm.graphics.tsa.plot_acf(df['qx'], lags=50)

            trace_id = os.path.splitext(os.path.basename(trace_path))[0]
            dest = os.path.join(output_path, f"Fig-{trace_id}-{dataset_type}.pdf")
            plt.savefig(dest)
            logging.info("Autocorrelation trace {} and saving to file {}".format(trace_path, dest))

    @staticmethod
    def plot_average(dataset_path, output_path, dataset_type):
        out_dir_df = "./data/average/"
        av_time = 1000
        for trace_path in utils.get_csv_files(dataset_path):
            df = pd.read_csv(trace_path, skipfooter=1, engine='python')
            df = df.rolling(av_time).mean()

            ts = np.arange(0, dataset_lengh_sec + cfg['dt'], cfg['dt'])

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

            # Plot position
            ax1.plot(ts, df.loc[:len(ts) - 1, 'x'], label='x')
            ax1.plot(ts, df.loc[:len(ts) - 1, 'y'], label='y', linestyle='--')
            ax1.plot(ts, df.loc[:len(ts) - 1, 'z'], label='z', linestyle='-.')
            ax1.set_ylabel('meters')
            ax1.set_xlim(0, dataset_lengh_sec)
            ax1.legend(loc='upper left')
            ax1.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
            ax1.xaxis.set_major_locator(MultipleLocator(10))

            # Plot orientation in Quaternions
            '''
            Q =  [qx, qy, qz, qw] = qv + qw, where 
            qw is the real part  and 
            qv = iqx + jqy + kqz= (qx, qy, qz) 
            is the imaginary part 
            x, y and z represent a vector. w is a scalar that stores the rotation around the vector.
            '''
            ax2.plot(ts, df.loc[:len(ts) - 1, 'qx'], label='qx', linestyle='solid')
            ax2.plot(ts, df.loc[:len(ts) - 1, 'qy'], label='qy', linestyle='--')
            ax2.plot(ts, df.loc[:len(ts) - 1, 'qz'], label='qz', linestyle='-.')
            ax2.plot(ts, df.loc[:len(ts) - 1, 'qw'], label='real qw', linestyle='solid')
            ax2.set_xlabel('seconds')
            ax2.set_ylabel('degrees')
            ax2.set_xlim(0, dataset_lengh_sec)
            ax2.legend(loc='upper left')
            ax2.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
            ax2.xaxis.set_major_locator(MultipleLocator(10))

            trace_id = os.path.splitext(os.path.basename(trace_path))[0]
            dest = os.path.join(output_path, f"Fig-{trace_id}-{dataset_type}.pdf")
            fig.savefig(dest)
            logging.info("Plotting trace {} and saving to file {}".format(trace_path, dest))

            case = os.path.splitext(os.path.basename(trace_path))[0]
            df_out = df.drop(labels=range(0, 300), axis=0)
            if not os.path.exists(out_dir_df):
                os.makedirs(out_dir_df)
            df_out.to_csv(os.path.join(out_dir_df, case + '.csv'), index=False)


    @staticmethod
    def plot_corr_matrix(dataset_path, output_path, column, dataset_type):
        for trace_path in utils.get_csv_files(dataset_path):
            df = pd.read_csv(trace_path, skipfooter=1, engine='python')
            correlation_matrix = df.drop([column], axis=1).corr()
            print(correlation_matrix)

            fig, ax = plt.subplots(figsize=(13,10))

            sns.heatmap(correlation_matrix,
                        annot=True,
                        fmt='.2f',
                        #cmap='YlGnBu',
                        ax=ax);
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14);

            trace_id = os.path.splitext(os.path.basename(trace_path))[0]
            dest = os.path.join(output_path, f"Fig-{trace_id}-{dataset_type}.pdf")
            fig.savefig(dest)
            logging.info("Plotting trace {} and saving to file {}".format(trace_path, dest))

    @staticmethod
    def plot_hist(dataset_path, output_path, column, dataset_type):
        for trace_path in utils.get_csv_files(dataset_path):
            df = pd.read_csv(trace_path, skipfooter=1, engine='python')

            df[column].hist(bins=100)

            trace_id = os.path.splitext(os.path.basename(trace_path))[0]
            dest = os.path.join(output_path, f"Fig-{trace_id}-{dataset_type}_{column}.pdf")
            plt.savefig(dest)
            logging.info("Plotting trace {} and saving to file {}".format(trace_path, dest))

    @staticmethod
    def plot_losses(train_losses, val_losses, params, results_path):
        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Train/Val losses: {int(params['LAT']*1e3)}ms "
                  f"hidden: {params['hidden_dim']}, batch: "
                  f"{params['batch_size']}, dropout: {params['dropout']}, "
                  f"layers: {params['layers']}")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        #plt.show()

        head, _ = os.path.split(results_path)
        out = os.path.join(head, 'losses')
        if not os.path.exists(out):
            os.makedirs(out)
        dest = os.path.join(out, f"Fig-LAT{int(params['LAT']*1e3)}_"
                            f"hid{params['hidden_dim']}_epochs{params['epochs']}_"
                            f"batch{params['batch_size']}_drop{params['dropout']}_"
                            f"layers{params['layers']}.pdf")
        plt.savefig(dest)
        logging.info(f"Saved to file {dest}")

        plt.close()


