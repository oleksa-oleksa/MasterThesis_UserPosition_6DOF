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
import numpy as np
import pandas as pd
import toml
import torch
import torch.nn as nn
import torch.optim as optim
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from UserPrediction6DOF.models.lstm import LSTMModel1, LSTMModel2, LSTMModel3, LSTMModel4
from UserPrediction6DOF.models.gru import GRUModel1, GRUModel3, GRUModel31, GRUModel32, GRUModel33, GRUModel34, GRUModel35
from UserPrediction6DOF.models.lstm_fcn import LSTMFCNModel1
from .nn_trainer import NNTrainer
from scipy.linalg import block_diag
from .evaluator import Evaluator, DeepLearnEvaluator
from UserPrediction6DOF.tools import dataset_tools, utils
from .plotter import DataPlotter
from torchinfo import summary

cuda_path = "/mnt/output"
config_path = "UserPrediction6DOF/tools/config.toml"
job_id = os.path.basename(os.path.normpath(cuda_path))

# For more readable printing
np.set_printoptions(precision=6, suppress=True, linewidth=np.inf)


class RNNRunner:
    """Runs the RNN predictor with specified model (LSTM, GRU...)

    Predicts the next values, X(t+n), from the previous n observations Xt, X+1, …, and X(t+n-1).

    D - the input vector is a HoloLens data coming from log csv-file
    as input vector x_t of length 10 at each time step

    H - length of the hidden state H, the summary of the history so far.
    can be modified for obtaining different prediction results

    B - is a batch size, the number of training examples utilized in one iteration

    [D x B] is input_dim parameter in LSTM => [10 x B]
    [H x B] is hidden_dim parameter in LSTM ==> [100 x B]

    These two are will be concatenated (ie x_t is tacked onto the end of h_t),
    making a new input hx_t with dimension [(H + D ) x B].
    In this case, hx_t is now [100 + 10 ] x B = [110 x B]

    When this new input hx_t is introduced to the gates,
    the weights are of dimension [H x (H + D)], in our case [100 x 110].
    B isn’t here, so different batch sizes will not affect the results of LSTM output

    Matrix multiply uses hx_t and the weights of each gate, [H x (H + D)] * [(H + D) x B],
    which in our case is [100 x (100 + 10)] * [(100 + 10) x B] => [100 x 110] * [110 x B]

    This produces  [H x B] again ( 100 x B) and this is the dimension of the hidden vector
    and the cell state that will be passed onto the next step.

    """

    def __init__(self, model_name, pred_window, dataset_path, results_path):
        # -----  PRESET ----------#
        self.cuda = torch.cuda.is_available()
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        self.model_name = model_name
        self.pred_window = pred_window * 1e-3  # convert to seconds
        self.pred_step = int(self.pred_window / self.dt)
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.dists_path = None  # set by prepare_environment()
        self.model = None  # set by select_model()
        self.params = None  # set by select_model()
        self.prepare_raw_dataset = False
        self.prepare_test = False
        self.add_sliding_window = False
        self.load_before_split_with_sliding = False
        self.load_test_val_train_split_with_sliding = False
        self.split_train_test_with_sliding = True
        self.X, self.y = [], []
        self.X_w, self.y_w = [], []
        self.X_train, self.X_val, self.X_test = [], [], []
        self.y_train, self.y_val, self.y_test = [], [], []
        self.plotter = DataPlotter()

        # -------------  FEATURES ---------------#
        self.features = self.cfg['pos_coords'] + self.cfg['quat_coords'] + self.cfg['velocity']
        # only position and rotation
        # self.features = self.cfg['pos_coords'] + self.cfg['quat_coords']

        # --------------  OUTPUTS ---------------#
        # position and rotation in future will be predicted
        self.outputs = self.cfg['pos_coords'] + self.cfg['quat_coords']

        # ---------  MODEL HYPERPARAMETERS ----------#
        self.reducing_learning_rate = True  # decreases LR every ls_epochs for 70%
        self.learning_rate = 1e-4  # 1e-3 base Adam optimizer
        self.lr_epochs = 30
        self.lr_multiplicator = 0.5
        self.weight_decay = 1e-12  # 1e-6 base Adam optimizer

        # self.num_past = 20  # number of past time series to predict future
        self.input_dim = len(self.features)
        self.output_dim = len(self.outputs)  # 3 position parameter + 4 rotation parameter
        self.hidden_dim = 512  # number of features in hidden state
        self.batch_size = 128
        self.n_epochs = 500
        self.dropout = 0
        self.layer_dim = 1  # the number of RNN layers stacked on top of each other
        self.seq_length_input = 20  # input length of timeseries from the past
        self.seq_length_output = self.pred_step  # output length of timeseries in the future
        self.patience = 7
        self.delta = 0.005

        # -----  CREATE PYTORCH MODEL ----------#
        # prepare paths for environment
        self.prepare_paths()
        # sets hyperparameters if they are in os.environ
        self.set_model_hyperparams()
        # create a RNN model with hyperparameters
        self.create_model(model_name)

    def prepare_paths(self):
        self.dists_path = os.path.join(self.results_path, 'distances')
        if not os.path.exists(self.dists_path) and not self.cuda:
            os.makedirs(self.dists_path, exist_ok=True)

        # -----  CUDA FOR CPU FOR RUNNING IN CONTAINER ----------#
        # for running in Singularity container paths must be modified
        if self.cuda:
            self.results_path = os.path.join(cuda_path, 'job_results/tabular')
            # logging.info(f"Cuda true. results_path {self.results_path}")
            self.dists_path = os.path.join(self.results_path, 'distances')
            # logging.info(f"Cuda true. dists_path {self.dists_path}")

    def set_model_hyperparams(self):
        if 'RNN_PARAMETERS' in os.environ:
            logging.info("Using hyperparameters from os.environ")
            if 'HIDDEN_DIM' in os.environ:
                self.hidden_dim = int(os.getenv('HIDDEN_DIM'))
            if 'BATCH_SIZE' in os.environ:
                self.batch_size = int(os.getenv('BATCH_SIZE'))
            if 'N_EPOCHS' in os.environ:
                self.n_epochs = int(os.getenv('N_EPOCHS'))
            if 'DROPOUT' in os.environ:
                self.dropout = float(os.getenv('DROPOUT'))
            if 'LAYERS' in os.environ:
                self.layer_dim = int(os.getenv('LAYERS'))
            if 'LR_ADAM' in os.environ:
                self.learning_rate = float(os.getenv('LR_ADAM'))
            if 'WEIGHT_DECAY_ADAM' in os.environ:
                self.weight_decay = float(os.getenv('WEIGHT_DECAY_ADAM'))
            if 'LR_EPOCHS' in os.environ:
                self.lr_epochs = int(os.getenv('LR_EPOCHS'))
            if 'LR_MULTIPLICATOR' in os.environ:
                self.lr_multiplicator = float(os.getenv('LR_MULTIPLICATOR'))

    def create_model(self, model_name):
        # batch_first=True --> input is [batch_size, seq_len, input_size]
        # SELECTS MODEL
        if model_name == "lstm1":
            self.model = LSTMModel1(self.input_dim, self.hidden_dim,
                                    self.output_dim, self.dropout, self.layer_dim)
        elif model_name == "lstm2":
            self.model = LSTMModel2(self.seq_length_input, self.input_dim, self.hidden_dim,
                                    self.seq_length_output, self.output_dim,
                                    self.dropout, self.layer_dim)
        elif model_name == "lstm3":
            self.model = LSTMModel3(self.seq_length_input, self.input_dim, self.hidden_dim,
                                    self.seq_length_output, self.output_dim,
                                    self.dropout, self.layer_dim)
        elif model_name == "lstm4":
            self.model = LSTMModel4(self.seq_length_input, self.input_dim, self.hidden_dim,
                                    self.seq_length_output, self.output_dim,
                                    self.dropout, self.layer_dim)
        elif model_name == "lstm-fcn1":
            self.model = LSTMFCNModel1(self.input_dim, self.output_dim, self.seq_length_input)

        elif model_name == "gru1":
            self.model = GRUModel1(self.input_dim, self.hidden_dim,
                                   self.output_dim, self.dropout, self.layer_dim)

        elif model_name == "gru3":
            self.model = GRUModel3(self.input_dim, self.hidden_dim,
                                   self.output_dim, self.dropout, self.layer_dim)
        elif model_name == "gru31":
            self.model = GRUModel31(self.input_dim, self.hidden_dim, self.output_dim)

        elif model_name == "gru32":
            self.model = GRUModel32(self.input_dim, self.hidden_dim, self.output_dim)
        elif model_name == "gru33":
            self.model = GRUModel33(self.input_dim, self.hidden_dim, self.output_dim)
        elif model_name == "gru34":
            self.model = GRUModel34(self.input_dim, self.hidden_dim, self.output_dim)
        elif model_name == "gru35":
            self.model = GRUModel35(self.input_dim, self.hidden_dim, self.output_dim)
        self.params = {'LAT': self.pred_window[0], 'hidden_dim': self.hidden_dim, 'epochs': self.n_epochs,
                       'batch_size': self.batch_size, 'dropout': self.dropout, 'layers': self.layer_dim,
                       'model': model_name, 'seq_length_input': self.seq_length_input, 'lr': self.learning_rate,
                       'lr_reducing': self.reducing_learning_rate, 'lr_epochs': self.lr_epochs,
                       'weight_decay': self.weight_decay, 'patience': self.patience, 'delta': self.delta,
                       'lr_multiplicator': self.lr_multiplicator}

    def print_model_info(self):
        logging.info("----------------- Runing RNN Predictor ---------------------")
        logging.info(f"RNN model is {self.model.name}.")
        logging.info("Using VCA GPU Cluster ") if self.cuda else logging.info("Using hardware CPU")
        dict_items = self.params.items()

        first_line = list(dict_items)[:5]
        second_line = list(dict_items)[5:]
        result = ', '.join(str(key) + ': ' + str(value) for key, value in first_line)
        logging.info(result)
        result = ', '.join(str(key) + ': ' + str(value) for key, value in second_line)
        logging.info(result)

        logging.info(self.model)
        summary(self.model, input_size=(self.batch_size, self.seq_length_input, self.input_dim))

    def prepare_dataset(self, prepare_raw_dataset, prepare_test, add_sliding_window, load_before_split_with_sliding,
                        load_test_val_train_split_with_sliding, split_train_test_with_sliding, load_train_test_with_sliding):
        if prepare_raw_dataset:
            # Read full dataset from CSV file
            df = dataset_tools.load_dataset(self.dataset_path)
            # create 2D arrays of features and outputs
            self.X, self.y = dataset_tools.prepare_X_y(df, self.features, self.seq_length_input, self.pred_step, self.outputs)

        # prepare and save separate file for testing with Kalman and Baseline
        if prepare_test:
            df = dataset_tools.load_dataset(self.dataset_path)
            # short test if train-val-test was used
            df_test_slice = pd.DataFrame(data=df.iloc[96023:, :], columns=df.columns)
            # test without validation dataset
            # df_test_slice = pd.DataFrame(data=df.iloc[84006:, :], columns=df.columns)
            print(df_test_slice.shape)

            test_path = os.path.join(self.dataset_path, 'test')
            if not os.path.exists(test_path):
                os.makedirs(test_path, exist_ok=True)
            df_test_slice.to_csv(os.path.join(test_path, '1.csv'), index=False)
            logging.info('TEST 1.csv for Kalman and Baseline is created!')

        if add_sliding_window:
            # Features and outputs with sequence_len = sliding window
            self.X_w, self.y_w = dataset_tools.add_sliding_window(self.X, self.y, self.seq_length_input, self.pred_step)
            utils.save_numpy_array(self.dataset_path, 'X_w', self.X_w)
            utils.save_numpy_array(self.dataset_path, 'y_w', self.y_w)

        if load_before_split_with_sliding:
            self.X_w = utils.load_numpy_array(self.dataset_path, 'X_w')
            self.y_w = utils.load_numpy_array(self.dataset_path, 'y_w')

            # Splitting the data into train, validation, and test sets
            self.X_train, self.X_val, self.X_test, \
                self.y_train, self.y_val, self.y_test = dataset_tools.train_val_test_split(self.X_w, self.y_w, 0.2)

            logging.info(f"X_train {self.X_train.shape}, X_val {self.X_val.shape}, "
                         f"X_test{self.X_test.shape}, y_train {self.y_train.shape}, "
                         f"y_val {self.y_val.shape}, y_test {self.y_test.shape}")

            path = os.path.join(self.dataset_path, 'train_val_test')
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            utils.save_numpy_array(path, 'X_train', self.X_train)
            utils.save_numpy_array(path, 'X_val', self.X_val)
            utils.save_numpy_array(path, 'X_test', self.X_test)
            utils.save_numpy_array(path, 'y_train', self.y_train)
            utils.save_numpy_array(path, 'y_val', self.y_val)
            utils.save_numpy_array(path, 'y_test', self.y_test)

        if load_test_val_train_split_with_sliding:
            path = os.path.join(self.dataset_path, 'train_val_test')
            self.X_train = utils.load_numpy_array(path, 'X_train')
            self.X_val = utils.load_numpy_array(path, 'X_val')
            self.X_test = utils.load_numpy_array(path, 'X_test')
            self.y_train = utils.load_numpy_array(path, 'y_train')
            self.y_val = utils.load_numpy_array(path, 'y_val')
            self.y_test = utils.load_numpy_array(path, 'y_test')

        if split_train_test_with_sliding:
            self.X_w = utils.load_numpy_array(self.dataset_path, 'X_w')
            self.y_w = utils.load_numpy_array(self.dataset_path, 'y_w')

            # Splitting the data into train and test sets
            self.X_train, self.X_test, self.y_train, self.y_test = \
                dataset_tools.test_train_split(self.X_w, self.y_w, 0.3)

            logging.info(f"X_train {self.X_train.shape}, X_test{self.X_test.shape}, "
                         f"y_train {self.y_train.shape}, y_test {self.y_test.shape}")

            utils.save_numpy_array(self.dataset_path, 'X_train', self.X_train)
            utils.save_numpy_array(self.dataset_path, 'X_test', self.X_test)
            utils.save_numpy_array(self.dataset_path, 'y_train', self.y_train)
            utils.save_numpy_array(self.dataset_path, 'y_test', self.y_test)

        if load_train_test_with_sliding:
            self.X_train = utils.load_numpy_array(self.dataset_path, 'X_train')
            self.X_test = utils.load_numpy_array(self.dataset_path, 'X_test')
            self.y_train = utils.load_numpy_array(self.dataset_path, 'y_train')
            self.y_test = utils.load_numpy_array(self.dataset_path, 'y_test')

    # --------------- RUN RNN PREDICTOR --------------------- #
    def run(self):
        self.print_model_info()

        # preparing arrays for future initialization
        self.prepare_dataset(prepare_raw_dataset=False,
                             prepare_test=False,
                             add_sliding_window=False,
                             load_before_split_with_sliding=False,
                             load_test_val_train_split_with_sliding=True,
                             split_train_test_with_sliding=False,
                             load_train_test_with_sliding=False)

        # Mean Squared Error Loss Function
        # average of the squared differences between actual values and predicted values
        criterion = nn.MSELoss(reduction="mean")

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # load data with dataloaders
        train_loader, val_loader, test_loader, _ = dataset_tools.load_data(self.X_train, self.X_val, self.X_test,
                                                                                   self.y_train, self.y_val, self.y_test,
                                                                                   self.batch_size)
        # TRAIN WITH VALIDATION
        nn_train = NNTrainer(self.model, criterion, optimizer, self.params)

        nn_train.train(train_loader, val_loader, self.n_epochs)

        # PREDICTION ON TEST DATA
        logging.info('Training finished. Starting prediction on test data!')
        predictions, targets = nn_train.predict(test_loader, self.batch_size)
        # print(predictions.shape, targets.shape)

        # Compute evaluation metrics
        deep_eval = DeepLearnEvaluator(predictions, targets)
        deep_eval.eval_model()
        euc_dists = deep_eval.euc_dists
        ang_dists = np.rad2deg(deep_eval.ang_dists)

        np.save(os.path.join(self.dists_path,
                             'euc_dists_{}_{}ms.npy'.format(self.model.name, int(self.pred_window * 1e3))), euc_dists)
        np.save(os.path.join(self.dists_path,
                             'ang_dists_{}_{}ms.npy'.format(self.model.name, int(self.pred_window * 1e3))), ang_dists)

        logging.info("--------------------------------------------------------------")
        df_results = pd.DataFrame({'Trace': "dataset", 'LAT' : self.pred_window,
                                   'mae_euc': deep_eval.metrics['mae_euc'], 'mae_ang': deep_eval.metrics['mae_ang'],
                                   'rmse_euc': deep_eval.metrics['rmse_euc'], 'rmse_ang': deep_eval.metrics['rmse_ang']})
        df_results.to_csv(os.path.join(self.results_path, 'res_lstm.csv'), index=False)

        # LOGGING AND PLOTTING
        pred_dic = {'MAE_pos': deep_eval.metrics['mae_euc']}

        # Plot train and validation losses
        # self.plotter.plot_losses(nn_train.train_losses, nn_train.val_losses, self.params, pred_dic, self.results_path)

        # log train/val losses as csv-files
        utils.log_losses(nn_train.train_losses, nn_train.val_losses, self.model_name, self.params, pred_dic)
        # log model parameters
        utils.log_parameters(df_results, self.params)
        # log predicted values and targets
        utils.log_predictions(predictions, self.model_name, self.params, pred_dic)
        # log_targets(targets, self.model_name)


class BaselineRunner:
    """Runs the baseline no-prediction case over all traces"""
    def __init__(self, pred_window, dataset_path, results_path):
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        self.pred_window = pred_window * 1e-3   # convert to seconds
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.coords = self.cfg['pos_coords'] + self.cfg['quat_coords']
        
    def run(self):
        logging.info("Baseline (no-prediction)")
        results = []
        
        for trace_path in utils.get_csv_files(self.dataset_path):
            basename = os.path.splitext(os.path.basename(trace_path))[0]
            logging.info("-------------------------------------------------------------------------")
            logging.info("Trace path: %s", trace_path)
            logging.info("-------------------------------------------------------------------------")
            for w in self.pred_window:
                logging.info("Prediction window = %s ms", w * 1e3)
                
                # Read trace from CSV file
                df_trace = pd.read_csv(trace_path)
                data = df_trace[self.coords].to_numpy()
                
                pred_step = int(w / self.dt)
                targets = data[pred_step:, :]   # Assumption: LAT = E2E latency
                preditions = data[:-pred_step, :]

                # create csv-file that can be to be used for plotting
                # utils.log_predictions(preditions[:, :7], 'baseline')
                
                # Compute evaluation metrics
                eval = Evaluator(targets, preditions, pred_step)
                eval.eval_baseline()
                metrics = np.array(list(eval.metrics.values()))
                result_one_experiment = list(np.hstack((basename, w, metrics)))
                results.append(result_one_experiment)
                logging.info("--------------------------------------------------------------")
        
        df_results = pd.DataFrame(results, columns=['Trace', 'LAT', 'mae_euc', 'mae_ang',
                                                    'rmse_euc', 'rmse_ang'])
        df_results.to_csv(os.path.join(self.results_path, 'res_baseline.csv'), index=False)
        
        
class KalmanRunner:
    """Runs the Kalman predictor over all traces"""
    def __init__(self, pred_window, dataset_path, results_path):
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
        # print(x_preds[:, ::2])
        # print(x_preds.shape)
        # create csv-file that can be to be used for plotting
        # utils.log_predictions(x_preds[:, ::2], 'kalman')
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
        
        for trace_path in utils.get_csv_files(self.dataset_path):
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




