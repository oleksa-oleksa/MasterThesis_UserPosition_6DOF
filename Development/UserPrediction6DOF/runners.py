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
import torch.optim as optim
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from .lstm import LSTMModel, LSTMOptimization
from scipy.linalg import block_diag
from statsmodels.iolib.smpickle import save_pickle
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults, ar_select_order
from .evaluator import Evaluator, DeepLearnEvaluator
from .utils import *

cuda_path = "/mnt/output"
job_id = os.path.basename(os.path.normpath(cuda_path))

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
            logging.info("-------------------------------------------------------------------------")
            logging.info("Trace path: %s", trace_path)
            logging.info("-------------------------------------------------------------------------")
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
                logging.info("--------------------------------------------------------------")
        
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
            os.makedirs(dists_path, exist_ok=True)
        
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

        ## TODO: Prepare dataset for dataloader
        ## TODO: and try the custom LSTM

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

                # Compute evaluation metrics BASELINE
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
        # -----  PRESET ----------#
        config_path = os.path.join(os.getcwd(), 'config.toml')
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        if 'RNN_PARAMETERS' in os.environ:
            self.pred_window = int(os.getenv('PRED_WINDOW')) * 1e-3  # convert to seconds
        else:
            self.pred_window = 100 * 1e-3  # convert to seconds

        self.dataset_path = dataset_path
        self.results_path = results_path
        self.dists_path = os.path.join(self.results_path, 'distances')

        # -----  CUDA FOR CPU ----------#
        # for running in Singularity container paths must be modified
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.results_path = os.path.join(cuda_path, 'job_results/tabular')
            logging.info(f"Cuda true. results_path {self.results_path}")
            self.dists_path = os.path.join(self.results_path, 'distances')
            logging.info(f"Cuda true. dists_path {self.dists_path}")

        # -----  FEATURES ----------#
        # features with velocity
        self.features = self.cfg['pos_coords'] + self.cfg['quat_coords'] + self.cfg['velocity']
        # only position and rotation without velocity and speed
        # self.features = self.cfg['pos_coords'] + self.cfg['quat_coords']

        # -----  MODEL HYPERPARAMETERS ----------#
        self.input_dim = 10  # 11 features with velocity and speed
        self.layer_dim = 1  # the number of LSTM layers stacked on top of each other
        self.output_dim = 7  # 3 position parameter + 4 rotation parameter
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6

        if 'RNN_PARAMETERS' in os.environ:
            self.hidden_dim = int(os.getenv('HIDDEN_DIM'))
            self.batch_size = int(os.getenv('BATCH_SIZE'))
            self.n_epochs = int(os.getenv('N_EPOCHS'))
            self.dropout = float(os.getenv('DROPOUT'))
        else:
            self.hidden_dim = 2
            self.batch_size = 1024
            self.n_epochs = 10
            self.dropout = 0.2

        # -----  CREATE PYTORH MODEL ----------#
        # input_dim, hidden_dim, layer_dim, output_dim, dropout_prob
        # batch_first=True --> input is [batch_size, seq_len, input_size]
        self.model = LSTMModel(self.input_dim, self.hidden_dim, self.output_dim, self.layer_dim)

    def run(self):
        logging.info(f"LSTM Base: hidden_dim: {self.hidden_dim}, batch_size: {self.batch_size}, "
                     f"n_epochs: {self.n_epochs}, dropout: {self.dropout}")
        results = []
        if not os.path.exists(self.dists_path):
            os.makedirs(self.dists_path, exist_ok=True)

        for trace_path in get_csv_files(self.dataset_path):
            basename = os.path.splitext(os.path.basename(trace_path))[0]
            logging.info("-------------------------------------------------------------------------")
            logging.info("Trace path: %s", trace_path)
            logging.info("-------------------------------------------------------------------------")
            for w in self.pred_window:
                logging.info("Prediction window = %s ms", w * 1e3)

                # Read trace from CSV file
                df_trace = pd.read_csv(trace_path)
                X = df_trace[self.features].to_numpy() # features.shape (12001, 11)

                pred_step = int(w / self.dt)

                # output is created from the features shifted corresponding to given latency
                y = X[pred_step:, :]  # Assumption: LAT = E2E latency
                # labels.shape
                # 20 ms (11997, 11) => 12001 - 20/5
                # 100 ms (11981, 11) => 12002 - 100/5

                # prepare features and labels
                X_cut = cut_dataset_lenght(X, y)
                y_cut = cut_extra_labels(y)

                # Splitting the data into train, validation, and test sets
                X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_cut, y_cut, 0.2)

                logging.info(f"X_train {X_train.shape}, X_val {X_val.shape}, X_test{X_test.shape}, "
                             f"y_train {y_train.shape}, y_val {y_val.shape}, y_test {y_test.shape}")

                train_loader, val_loader, test_loader, test_loader_one = load_data(X_train, X_val, X_test,
                                                                  y_train, y_val, y_test, batch_size=self.batch_size)

                # Long Short-Term Memory TRAIN + EVAL

                # Mean Squared Error Loss Function
                # average of the squared differences between actual values and predicted values
                loss_fn = nn.MSELoss(reduction="mean")

                # Mean Absolute Error (L1 Loss Function)
                # average of the sum of absolute differences between actual values and predicted values
                # loss_fn = nn.L1Loss()
                optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

                opt = LSTMOptimization(model=self.model, loss_fn=loss_fn, optimizer=optimizer)
                opt.train(train_loader, val_loader, batch_size=self.batch_size, n_epochs=self.n_epochs, n_features=self.input_dim)
                # opt.plot_losses()

                predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=self.input_dim)

                # predictions.shape is [(2400, 1, 7)]
                # Remove axes of length one from predictions.
                predictions = np.array(predictions).squeeze()
                values = np.array(values).squeeze()

                # Debug info
                # print_result(predictions, values)
                # print(f"y_test is close to values? {np.allclose(y_test, values, atol=1e-08)}")

                # Compute evaluation metrics LSTM
                deep_eval = DeepLearnEvaluator(predictions, values)
                deep_eval.eval_lstm()
                metrics = np.array(list(deep_eval.metrics.values()))
                euc_dists = deep_eval.euc_dists
                ang_dists = np.rad2deg(deep_eval.ang_dists)

                np.save(os.path.join(self.dists_path,
                                     'euc_dists_lstm_{}_{}ms.npy'.format(basename, int(w * 1e3))), euc_dists)
                np.save(os.path.join(self.dists_path,
                                     'ang_dists_lstm_{}_{}ms.npy'.format(basename, int(w * 1e3))), ang_dists)

                result_single = list(np.hstack((basename, w, metrics)))
                results.append(result_single)

                logging.info("--------------------------------------------------------------")

        df_results = pd.DataFrame(results, columns=['Trace', 'LAT', 'mae_euc', 'mae_ang',
                                                    'rmse_euc', 'rmse_ang'])
        df_results.to_csv(os.path.join(self.results_path, 'res_lstm.csv'), index=False)

        # log model parameters
        log_parameters(self.hidden_dim, self.n_epochs, self.batch_size, self.dropout, df_results)
