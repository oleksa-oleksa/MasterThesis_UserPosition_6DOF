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
        zs_pos = self.zs[:-self.pred_step, :3]
        zs_rot = self.zs[:-self.pred_step:, 3:]
        zs_rot = np.array([Quaternion(q) for q in zs_rot])
        
        zs_shifted_pos = self.preds[:, :3]
        zs_shifted_rot = self.preds[:, 3:]
        zs_shifted_rot = np.array([Quaternion(q) for q in zs_shifted_rot])
    
        self.compute_metrics(zs_pos, zs_rot, zs_shifted_pos, zs_shifted_rot)


class DeepLearnEvaluator():
    """Compute evaluation metrics MAE and RMSE for different predictors"""
    def __init__(self, predictions, actual_values):
        self.predictions = predictions
        self.actual_values = actual_values
        self.euc_dists = None
        self.ang_dists = None
        self.metrics = {}

    def eval_model(self):
        """
        New HoloLens data of length 11:
        [x, y, z] => [:, :3] three first columns
        [qw, qx, qy, qz] => [:, 3:7] next 4 columns
        the rest of vector is velocity + speed data that will not (!) be predicted and evaluated
        """

        # split predictions array into position and rotations
        preds_pos = self.predictions[:, :3] # x, y, z
        preds_rot = self.predictions[:, 3:7] # qx, qy, qz, qw
        preds_rot = np.array([Quaternion(q) for q in preds_rot])

        actual_pos = self.actual_values[:, :3]
        actual_rot = self.actual_values[:, 3:7]
        actual_rot = np.array([Quaternion(q) for q in actual_rot])

        self.compute_metrics(preds_pos, preds_rot, actual_pos, actual_rot)

    def compute_metrics(self, preds_pos, preds_rot, actual_pos, actual_rot):
        """
        Based on a rule of thumb, RMSE values between 0.2 and 0.5
        show that the model can relatively predict the data accurately.
        """
        # Compute Eucliden and angular distances
        self.euc_dists = np.linalg.norm(preds_pos - actual_pos, axis=1)
        self.ang_dists = np.array([Quaternion.distance(q1, q2) for q1, q2 in zip(preds_rot,
                                                                                 actual_rot)])

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

        """
        result_metrics = {'mae': mean_absolute_error(df.value, df.prediction),
                          'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
                          'r2': r2_score(df.value, df.prediction)}

        print("Mean Absolute Error:       ", result_metrics["mae"])
        print("Root Mean Squared Error:   ", result_metrics["rmse"])
        print("R^2 Score:                 ", result_metrics["r2"])
        return result_metrics
        """