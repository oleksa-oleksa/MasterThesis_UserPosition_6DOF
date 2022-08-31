import toml
from UserPrediction6DOF.tools import dataset_tools, utils
import logging
import os
from statistics import mean, stdev

config_path = "UserPrediction6DOF/tools/config.toml"


class Normalization:
    def __init__(self, features, outputs, dataset_path, results_path, pred_window=100):
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        self.pred_window = pred_window * 1e-3  # convert to seconds
        self.pred_step = int(self.pred_window / self.dt)
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.X, self.y = [], []
        self.X_w, self.y_w = [], []
        self.X_train, self.X_val, self.X_test = [], [], []
        self.y_train, self.y_val, self.y_test = [], [], []
        self.config = None
        self.seq_length_input = 20  # input length of timeseries from the past
        self.features = features
        self.outputs = outputs
        self.min = []
        self.max = []
        self.mean = []
        self.min_dic = {}
        self.max_dic = {}
        self.mean_dic = {}

    def _prepare_raw_dataset(self):
        # Read full dataset from CSV file
        df = dataset_tools.load_dataset(self.dataset_path)
        # create 2D arrays of features and outputs
        self.X, self.y = dataset_tools.prepare_X_y(df, self.features, self.seq_length_input, self.pred_step, self.outputs)

    def _normalize_dataset(self, norm_type):

        for i in range(3):
            self.mean.append(mean(self.X[:, i]))
            self.min.append(min(self.X[:, i]))
            self.max.append(max(self.X[:, i]))

        self.param_dic = {
            'x_mean': self.mean[0], 'y_mean': self.mean[1], 'z_mean': self.mean[2],
            'x_min': self.min[0], 'y_min': self.min[1], 'z_min': self.min[2],
            'x_max': self.max[0], 'y_max': self.max[1], 'z_max': self.max[2]
        }

        if norm_type == "mean":
            self.X[:, 0] = (self.X[:, 0] - mean(self.X[:, 0])) / stdev(self.X[:, 0])
            self.X[:, 1] = (self.X[:, 1] - mean(self.X[:, 1])) / stdev(self.X[:, 1])
            self.X[:, 2] = (self.X[:, 2] - mean(self.X[:, 2])) / stdev(self.X[:, 2])

            self.y[:, 0] = (self.y[:, 0] - mean(self.y[:, 0])) / stdev(self.y[:, 0])
            self.y[:, 1] = (self.y[:, 1] - mean(self.y[:, 1])) / stdev(self.y[:, 1])
            self.y[:, 2] = (self.y[:, 2] - mean(self.y[:, 2])) / stdev(self.y[:, 2])

        elif norm_type == 'min-max':
            self.X[:, 0] = (self.X[:, 0] - min(self.X[:, 0])) / (max(self.X[:, 0]) - min(self.X[:, 0]))
            self.X[:, 1] = (self.X[:, 1] - min(self.X[:, 1])) / (max(self.X[:, 1]) - min(self.X[:, 1]))
            self.X[:, 2] = (self.X[:, 2] - min(self.X[:, 2])) / (max(self.X[:, 2]) - min(self.X[:, 2]))

            self.y[:, 0] = (self.y[:, 0] - min(self.y[:, 0])) / (max(self.y[:, 0]) - min(self.y[:, 0]))
            self.y[:, 1] = (self.y[:, 1] - min(self.y[:, 1])) / (max(self.y[:, 1]) - min(self.y[:, 1]))
            self.y[:, 2] = (self.y[:, 2] - min(self.y[:, 2])) / (max(self.y[:, 2]) - min(self.y[:, 2]))

        elif norm_type == 'min-max-double':
            self.X[:, 0] = 2 * ((self.X[:, 0] - min(self.X[:, 0])) / (max(self.X[:, 0]) - min(self.X[:, 0]))) - 1
            self.X[:, 1] = 2 * ((self.X[:, 1] - min(self.X[:, 1])) / (max(self.X[:, 1]) - min(self.X[:, 1]))) - 1
            self.X[:, 2] = 2 * ((self.X[:, 2] - min(self.X[:, 2])) / (max(self.X[:, 2]) - min(self.X[:, 2]))) - 1

            self.y[:, 0] = 2 * ((self.y[:, 0] - min(self.y[:, 0])) / (max(self.y[:, 0]) - min(self.y[:, 0]))) - 1
            self.y[:, 1] = 2 * ((self.y[:, 1] - min(self.y[:, 1])) / (max(self.y[:, 1]) - min(self.y[:, 1]))) - 1
            self.y[:, 2] = 2 * ((self.y[:, 2] - min(self.y[:, 2])) / (max(self.y[:, 2]) - min(self.y[:, 2]))) - 1

    def _add_sliding_window(self):
        # Features and outputs with sequence_len = sliding window
        self.X_w, self.y_w = dataset_tools.add_sliding_window(self.X, self.y, self.seq_length_input, self.pred_step)

    def _split(self):
        # Splitting the data into train, validation, and test sets
        self.X_train, self.X_val, self.X_test, \
        self.y_train, self.y_val, self.y_test = dataset_tools.train_val_test_split(self.X_w, self.y_w, 0.2)

        logging.info(f"X_train {self.X_train.shape}, X_val {self.X_val.shape}, "
                     f"X_test{self.X_test.shape}, y_train {self.y_train.shape}, "
                     f"y_val {self.y_val.shape}, y_test {self.y_test.shape}")

        path = os.path.join(self.results_path, 'train_val_test')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        utils.save_numpy_array(path, 'X_train', self.X_train)
        utils.save_numpy_array(path, 'X_val', self.X_val)
        utils.save_numpy_array(path, 'X_test', self.X_test)
        utils.save_numpy_array(path, 'y_train', self.y_train)
        utils.save_numpy_array(path, 'y_val', self.y_val)
        utils.save_numpy_array(path, 'y_test', self.y_test)

    def _log_reverse_parameters(self):
        output_file_name = "reverse_parameters.toml"
        path = os.path.join(self.results_path, 'train_val_test')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        log_path = os.path.join(path, output_file_name)

        with open(log_path, "w+") as toml_file:
            toml.dump(self.param_dic, toml_file)
