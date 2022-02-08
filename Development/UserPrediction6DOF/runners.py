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

import logging
import os
import pickle
from math import floor
import numpy as np
import pandas as pd
import toml
import torch
import torch.nn as nn
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from statsmodels.iolib.smpickle import save_pickle
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults, ar_select_order
from .evaluator import Evaluator
from .utils import get_csv_files

# For more readable printing
np.set_printoptions(precision=6, suppress=True, linewidth=np.inf)


class BaselineRunner():
    """Runs the baseline no-prediction case over all traces"""
    def __init__(self, pred_window, dataset_path, results_path):
        config_path = os.path.join(os.getcwd(), 'config.toml')
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        self.pred_window = pred_window * 1e-3   # convert to seconds
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.coords = self.cfg['pos_coords'] + self.cfg['quat_coords']
        
    def run(self):
        logging.info("Baseline (no-prediction)")
        results = []
        
        for trace_path in get_csv_files(self.dataset_path):
            basename = os.path.splitext(os.path.basename(trace_path))[0]
            print("-------------------------------------------------------------------------")
            logging.info("Trace path: %s", trace_path)
            print("-------------------------------------------------------------------------")
            for w in self.pred_window:
                logging.info("Prediction window = %s ms", w * 1e3)
                
                # Read trace from CSV file
                df_trace = pd.read_csv(trace_path)
                zs = df_trace[self.coords].to_numpy()
                
                pred_step = int(w / self.dt)
                zs_shifted = zs[pred_step:, :]   # Assumption: LAT = E2E latency
                
                # Compute evaluation metrics
                eval = Evaluator(zs, zs_shifted, pred_step)
                eval.eval_baseline()
                metrics = np.array(list(eval.metrics.values()))
                result_one_experiment = list(np.hstack((basename, w, metrics)))
                results.append(result_one_experiment)
                print("--------------------------------------------------------------")
        
        df_results = pd.DataFrame(results, columns=['Trace', 'LAT', 'mae_euc', 'mae_ang',
                                                    'rmse_euc', 'rmse_ang'])
        df_results.to_csv(os.path.join(self.results_path, 'res_baseline.csv'), index=False)
        
        
class KalmanRunner():
    """Runs the Kalman predictor over all traces"""
    def __init__(self, pred_window, dataset_path, results_path):
        config_path = os.path.join(os.getcwd(), 'config.toml')
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        self.pred_window = pred_window * 1e-3   # convert to seconds
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.coords = self.cfg['pos_coords'] + self.cfg['quat_coords']
        self.kf = KalmanFilter(dim_x = self.cfg['dim_x'], dim_z = self.cfg['dim_z'])
        setattr(self.kf, 'x_pred', self.kf.x)

        # First-order motion model: insert dt into the diagonal blocks of F
        f = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.kf.F = block_diag(f, f, f, f, f, f, f)

        # Inserts 1 into the blocks of H to select the measuremetns
        np.put(self.kf.H, np.arange(0, self.kf.H.size, self.kf.dim_x + 2), 1.0)
        self.kf.R *= self.cfg['var_R']
        Q_pos = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.cfg['var_Q_pos'], block_size=3)
        Q_ang = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.cfg['var_Q_ang'], block_size=4)
        self.kf.Q = block_diag(Q_pos, Q_ang)

    def reset(self):
        logging.debug("Reset Kalman filter")
        self.kf.x = np.zeros((self.cfg['dim_x'], 1))
        self.kf.P = np.eye(self.cfg['dim_x'])

    def lookahead(self):
        self.kf.x_pred = np.dot(self.kf.F_lookahead, self.kf.x)

    def run_single(self, trace_path, w):
        # Adjust F depending on the lookahead time
        f_l = np.array([[1.0, w], [0.0, 1.0]])
        setattr(self.kf, 'F_lookahead', block_diag(f_l, f_l, f_l, f_l, f_l, f_l, f_l))

        # Read trace from CSV file
        df_trace = pd.read_csv(trace_path)
        xs, covs, x_preds = [], [], []
        zs = df_trace[self.coords].to_numpy()
        z_prev = np.zeros(7)
        for z in zs:
            sign_array = -np.sign(z_prev[3:]) * np.sign(z[3:])
            sign_flipped = all(e == 1 for e in sign_array)
            if sign_flipped:
                logging.debug("A sign flip occurred.")
                self.reset()
            self.kf.predict()
            self.kf.update(z)
            self.lookahead()
            xs.append(self.kf.x)
            covs.append(self.kf.P)
            x_preds.append(self.kf.x_pred)
            z_prev = z
        
        # Compute evaluation metrics
        xs = np.array(xs).squeeze()
        covs = np.array(covs).squeeze()
        x_preds = np.array(x_preds).squeeze()
        pred_step = int(w / self.dt)
        eval = Evaluator(zs, x_preds[:, ::2], pred_step)
        eval.eval_kalman()
        metrics = np.array(list(eval.metrics.values()))
        euc_dists = eval.euc_dists
        ang_dists = np.rad2deg(eval.ang_dists)
        
        return metrics, euc_dists, ang_dists
    
    def run(self):
        logging.info("Kalman filter")
        results = []
        
        dists_path = os.path.join(self.results_path, 'distances')
        if not os.path.exists(dists_path):
            os.makedirs(dists_path)
        
        for trace_path in get_csv_files(self.dataset_path):
            basename = os.path.splitext(os.path.basename(trace_path))[0]
            print("-------------------------------------------------------------------------")
            logging.info("Trace path: %s", trace_path)
            print("-------------------------------------------------------------------------")
            for w in self.pred_window:
                logging.info("Prediction window = %s ms", w * 1e3)
                self.reset()

                metrics, euc_dists, ang_dists = self.run_single(trace_path, w)
                np.save(os.path.join(dists_path, 
                                        'euc_dists_{}_{}ms.npy'.format(basename, int(w*1e3))), euc_dists)
                np.save(os.path.join(dists_path, 
                                        'ang_dists_{}_{}ms.npy'.format(basename, int(w*1e3))), ang_dists)
                result_single = list(np.hstack((basename, w, metrics)))
                results.append(result_single)
                print("--------------------------------------------------------------")

        # Save metrics
        df_results = pd.DataFrame(results, columns=['Trace', 'LAT', 'mae_euc', 'mae_ang',
                                                    'rmse_euc', 'rmse_ang'])
        df_results.to_csv(os.path.join(self.results_path, 'res_kalman.csv'), index=False)


class LSTMBaseRunner():
    """Runs the LSTM NN over all traces

    Predicts the next value, X(t+n), from the previous n observations Xt, X+1, …, and X(t+n-1).
    """

    def __init__(self, pred_window, dataset_path, results_path):
        config_path = os.path.join(os.getcwd(), 'config.toml')
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        self.pred_window = pred_window * 1e-3  # convert to seconds
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.features = self.cfg['pos_coords'] + self.cfg['quat_coords'] + self.cfg['velocity'] + self.cfg['speed']

        ## TODO: Prepare dataset for dataloader with tutorial
        ## https://github.com/piEsposito/pytorch-lstm-by-hand/blob/master/nlp-naive-lstm-byhand.ipynb

    def run(self):
        logging.info("LSTM Base (Long Short-Term Memory Network)")
        results = []

        for trace_path in get_csv_files(self.dataset_path):
            basename = os.path.splitext(os.path.basename(trace_path))[0]
            print("-------------------------------------------------------------------------")
            logging.info("Trace path: %s", trace_path)
            print("-------------------------------------------------------------------------")
            for w in self.pred_window:
                logging.info("Prediction window = %s ms", w * 1e3)

                # Read trace from CSV file
                df_trace = pd.read_csv(trace_path)
                features = df_trace[self.features].to_numpy()

                pred_step = int(w / self.dt)

                # output is created from the features shifted corresponding to given latency
                labels = features[pred_step:, :]  # Assumption: LAT = E2E latency

                # Compute evaluation metrics
                evaluator = Evaluator(features, labels, pred_step)
                evaluator.eval_lstm_base()
                metrics = np.array(list(evaluator.metrics.values()))
                result_one_experiment = list(np.hstack((basename, w, metrics)))
                results.append(result_one_experiment)
                print("--------------------------------------------------------------")

        df_results = pd.DataFrame(results, columns=['Trace', 'LAT', 'mae_euc', 'mae_ang',
                                                    'rmse_euc', 'rmse_ang'])
        df_results.to_csv(os.path.join(self.results_path, 'res_lstm_base.csv'), index=False)


class LSTMRunner():
    """Runs the LSTM NN over all traces

    Predicts the next value, X(t+n), from the previous n observations Xt, X+1, …, and X(t+n-1).

    D - the input vector is a HoloLens data coming from log csv-file
    as input vector x_t of length 11 at each time step

    H - length of the hidden state H, the summary of the history so far.
    can be modified for obtaining different prediction results

    B - is a batch size, the number of training examples utilized in one iteration

    [D x B] is input_dim parameter in LSTM => [11 x B]
    [H x B] is hidden_dim parameter in LSTM ==> [100 x B]

    These two are will be concatenated (ie x_t is tacked onto the end of h_t),
    making a new input hx_t with dimension [(H + D ) x B].
    In this case, hx_t is now [100 + 11 ] x B = [111 x B]

    When this new input hx_t is introduced to the gates,
    the weights are of dimension [H x (H + D)], in our case [100 x 111].
    B isn’t here, so different batch sizes will not affect the results of LSTM output

    Matrix multiply uses hx_t and the weights of each gate, [H x (H + D)] * [(H + D) x B],
    which in our case is [100 x (100 + 11)] * [(100 + 11) x B] => [100 x 111] * [111 x B]
    50x55 * 55x1.

    This produces  [H x B] again ( 100 x B) and this is the dimension of the hidden vector
    and the cell state that will be passed onto the next step.

    """

    def __init__(self, pred_window, dataset_path, results_path):
        config_path = os.path.join(os.getcwd(), 'config.toml')
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        self.pred_window = pred_window * 1e-3  # convert to seconds
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.features = self.cfg['pos_coords'] + self.cfg['quat_coords'] + self.cfg['velocity'] + self.cfg['speed']

        ## TODO: Prepare dataset for dataloader with tutorial
        ## https://github.com/piEsposito/pytorch-lstm-by-hand/blob/master/nlp-naive-lstm-byhand.ipynb

    def run(self):
        logging.info("LSTM PyTorch (Long Short-Term Memory Network)")
        results = []

        for trace_path in get_csv_files(self.dataset_path):
            basename = os.path.splitext(os.path.basename(trace_path))[0]
            print("-------------------------------------------------------------------------")
            logging.info("Trace path: %s", trace_path)
            print("-------------------------------------------------------------------------")
            for w in self.pred_window:
                logging.info("Prediction window = %s ms", w * 1e3)

                # Read trace from CSV file
                df_trace = pd.read_csv(trace_path)
                features = df_trace[self.features].to_numpy()

                pred_step = int(w / self.dt)

                # output is created from the features shifted corresponding to given latency
                labels = features[pred_step:, :]  # Assumption: LAT = E2E latency

                # Compute evaluation metrics
                evaluator = Evaluator(features, labels, pred_step)
                evaluator.eval_lstm_base()
                metrics = np.array(list(evaluator.metrics.values()))
                result_one_experiment = list(np.hstack((basename, w, metrics)))
                results.append(result_one_experiment)
                print("--------------------------------------------------------------")

        df_results = pd.DataFrame(results, columns=['Trace', 'LAT', 'mae_euc', 'mae_ang',
                                                    'rmse_euc', 'rmse_ang'])
        df_results.to_csv(os.path.join(self.results_path, 'res_lstm.csv'), index=False)
