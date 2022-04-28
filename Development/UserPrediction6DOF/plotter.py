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
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.signal import savgol_filter
from .utils import get_csv_files

style_path = os.path.join(os.getcwd(), 'UserPrediction6DOF/style.json')
style = json.load(open(style_path))
config_path = os.path.join(os.getcwd(), 'config.toml')
cfg = toml.load(config_path)
dataset_lengh_sec = 600


class DataPlotter():
    """Plots interpolated dataset traces"""
    @staticmethod
    def plot_datasets(dataset_path, output_path, dataset_type):
        logging.info(f"Plotting from {dataset_path} and saving to {output_path}")
        for trace_path in get_csv_files(dataset_path):
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

            trace_id = os.path.splitext(os.path.basename(trace_path))[0]
            dest = os.path.join(output_path, f"Fig-{trace_id}-{dataset_type}.pdf")
            fig.savefig(dest)
            logging.info("Plotting trace {} and saving to file {}".format(trace_path, dest))


