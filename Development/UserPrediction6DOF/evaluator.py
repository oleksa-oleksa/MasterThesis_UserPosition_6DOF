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

import numpy as np
from pyquaternion import Quaternion
import logging
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# HoloLens CSV-Log parameter
pos_size = 3
rot_size = 4
eval_stop = pos_size + rot_size


class Evaluator():
    """Compute evaluation metrics MAE and RMSE for different predictors"""
    def __init__(self, zs, preds, pred_step):
        self.zs = zs
        self.preds = preds
        self.pred_step = pred_step
        self.euc_dists = None
        self.ang_dists = None
        self.metrics = {}
    
    def compute_metrics(self, zs_pos, zs_rot, preds_pos, preds_rot):
        # Compute Eucliden and angular distances
        self.euc_dists = np.linalg.norm(zs_pos - preds_pos, axis=1)
        self.ang_dists = np.array([Quaternion.distance(q1, q2) for q1, q2 in zip(zs_rot,
                                                                               preds_rot)])
        
        # Mean Absolute Error (MAE)
        self.metrics['mae_euc'] = np.sum(self.euc_dists) / self.euc_dists.shape[0]
        logging.info("MAE position = %s", self.metrics['mae_euc'])
        self.metrics['mae_ang'] = np.rad2deg(np.sum(self.ang_dists) / self.ang_dists.shape[0])
        logging.info("MAE rotation = %s", self.metrics['mae_ang'])

        # Root Mean Squared Error (RMSE)
        self.metrics['rmse_euc'] = np.sqrt((self.euc_dists ** 2).mean())
        logging.info("RMSE position = %s", self.metrics['rmse_euc'])
        self.metrics['rmse_ang'] = np.rad2deg(np.sqrt((self.ang_dists ** 2).mean()))
        logging.info("RMSE rotation = %s", self.metrics['rmse_ang'])

    def eval_kalman(self):
        zs_pos = self.zs[self.pred_step:, :3]
        zs_rot = self.zs[self.pred_step:, 3:]
        zs_rot = np.array([Quaternion(q) for q in zs_rot])

        preds_pos = self.preds[:-self.pred_step, :3]
        preds_rot = self.preds[:-self.pred_step:, 3:]
        preds_rot = np.array([Quaternion(q) for q in preds_rot])

        self.compute_metrics(zs_pos, zs_rot, preds_pos, preds_rot)

    def eval_baseline(self):
        """
        zs_pos = self.zs[:-self.pred_step, :3]
        zs_rot = self.zs[:-self.pred_step:, 3:]
        zs_rot = np.array([Quaternion(q) for q in zs_rot])
        
        zs_shifted_pos = self.preds[:, :3]
        zs_shifted_rot = self.preds[:, 3:]
        zs_shifted_rot = np.array([Quaternion(q) for q in zs_shifted_rot])
    
        self.compute_metrics(zs_pos, zs_rot, zs_shifted_pos, zs_shifted_rot)
        """

        # values, values_lagged

        values_pos = self.zs[:, :3]
        values_rot = self.zs[:, 3:]
        values_rot = np.array([Quaternion(q) for q in values_rot])

        values_lagged_pos = self.preds[:, :3]
        values_lagged_rot = self.preds[:, 3:]
        values_lagged_rot = np.array([Quaternion(q) for q in values_lagged_rot])

        self.compute_metrics(values_pos, values_rot, values_lagged_pos, values_lagged_rot)


class DeepLearnEvaluator():
    """Compute evaluation metrics MAE and RMSE for different predictors"""
    def __init__(self, predictions, actual_values, dataset_type):
        self.predictions = predictions
        self.actual_values = actual_values
        self.dataset_type = dataset_type
        self.euc_dists = None
        self.angular_dist = None
        self.geodesic_dist = None
        self.metrics = {}

    def eval_model(self):
        """
        New HoloLens data of length 11:
        [x, y, z] => [:, :3] three first columns
        [qw, qx, qy, qz] => [:, 3:7] next 4 columns
        the rest of vector is velocity + speed data that will not (!) be predicted and evaluated
        """
        preds_pos, preds_rot, actual_pos, actual_rot = [], [], [], []

        if self.dataset_type is None:
            logging.info(f'No dataset type!')

        elif self.dataset_type == 'full':
            # split predictions array into position and rotations
            preds_pos = self.predictions[:, :3] # x, y, z
            preds_rot = self.predictions[:, 3:7] # qx, qy, qz, qw
            preds_rot = np.array([Quaternion(q) for q in preds_rot])

            actual_pos = self.actual_values[:, :3]
            actual_rot = self.actual_values[:, 3:7]
            actual_rot = np.array([Quaternion(q) for q in actual_rot])

        elif self.dataset_type == 'position' or self.dataset_type == 'position_velocity':
            preds_pos = self.predictions  # x, y, z
            actual_pos = self.actual_values  # x, y, z

        elif self.dataset_type == 'rotation' or self.dataset_type == 'rotation_velocity':
            preds_rot = np.array([Quaternion(q) for q in self.predictions])  # qx, qy, qz, qw
            actual_rot = np.array([Quaternion(q) for q in self.actual_values])   # qx, qy, qz, qw

        self.compute_metrics(preds_pos, preds_rot, actual_pos, actual_rot)

    @staticmethod
    def calc_angular_dist(preds_rot, actual_rot):

        zs = [q1 * q2.conjugate for q1, q2 in zip(preds_rot, actual_rot)]
        reals = np.fromiter((z.real for z in zs), dtype=float)
        reals = reals[(reals >= -1) & (reals <=1)]
        theta = 2*np.arccos(np.abs(reals))
        logging.info(f'np.arccos() out or range [-1..1]: {preds_rot.shape[0] - theta.shape[0]} times!')
        return theta

    def compute_metrics(self, preds_pos, preds_rot, actual_pos, actual_rot):
        """
        Based on a rule of thumb, RMSE values between 0.2 and 0.5
        show that the model can relatively predict the data accurately.
        """
        if self.dataset_type == 'full':
            # Compute Eucliden and angular distances
            self.euc_dists = np.linalg.norm(preds_pos - actual_pos, axis=1)
            self.angular_dist = self.calc_angular_dist(preds_rot, actual_rot)
            self.geodesic_dist = np.array([Quaternion.distance(q1, q2) for q1, q2 in zip(preds_rot, actual_rot)])

        # Mean Absolute Error (MAE)
        self.metrics['mae_euc'] = np.sum(self.euc_dists) / self.euc_dists.shape[0]
        logging.info("MAE position = %s", self.metrics['mae_euc'])
        self.metrics['mae_ang'] = np.rad2deg(np.sum(self.angular_dist) / self.angular_dist.shape[0])
        logging.info("MAE rotation angular = %s", self.metrics['mae_ang'])
        self.metrics['mae_geo'] = np.rad2deg(np.sum(self.geodesic_dist) / self.geodesic_dist.shape[0])
        logging.info("MAE rotation geodesic = %s", self.metrics['mae_geo'])

        # Root Mean Squared Error (RMSE)
        self.metrics['rmse_euc'] = np.sqrt((self.euc_dists ** 2).mean())
        logging.info("RMSE position = %s", self.metrics['rmse_euc'])
        self.metrics['rmse_ang'] = np.rad2deg(np.sqrt((self.angular_dist ** 2).mean()))
        logging.info("RMSE rotation angular = %s", self.metrics['rmse_ang'])
        self.metrics['rmse_geo'] = np.rad2deg(np.sqrt((self.geodesic_dist ** 2).mean()))
        logging.info("RMSE rotation geodesic = %s", self.metrics['rmse_geo'])