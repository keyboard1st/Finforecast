# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

def get_criterion(loss = 'MSE'):
    if loss == 'MSE':
        criterion = nn.MSELoss()
    elif loss == 'MAPE':
        criterion = mape_loss()
    elif loss == 'MASE':
        criterion = mase_loss()
    elif loss == 'SMAPE':
        criterion = smape_loss()
    elif loss == "WMSE":
        criterion = wmse_loss()
    elif loss == "IC_loss_one":
        criterion = IC_loss_one()
    elif loss == "IC_loss_mse":
        criterion = IC_loss_mse()
    elif loss == 'MHMSE':
        criterion = mse_reg_mh()
    else:
        raise ValueError('loss只能是MSE, MAPE, MASE, SMAPE中的一个')
    return criterion


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

class wmse_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, forecast: t.Tensor, target: t.Tensor) -> t.float:
        loss = (forecast - target) ** 2
        w = (t.sort(target).indices + 1)/ target.shape[0]
        w_loss = w * loss
        return t.mean(w_loss)

class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

class IC_loss_double(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_lgb, pred_gru, y_true):
        mutualstacked =t.stack([pred_lgb, pred_gru], dim=0)
        mulualic = t.corrcoef(mutualstacked)[0][1]

        predstacked = t.stack([y_true, pred_gru], dim=0)
        predIC = t.corrcoef(predstacked)[0][1]
        return - divide_no_nan(predIC, mulualic)

class IC_loss_double_diff(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_lgb, pred_gru, y_true, lamda = 0.005):
        mutualstacked =t.stack([pred_lgb, pred_gru], dim=0)
        mulualic = t.corrcoef(mutualstacked)[0][1]

        predstacked = t.stack([y_true, pred_gru], dim=0)
        predIC = t.corrcoef(predstacked)[0][1]
        return -predIC + lamda * mulualic

class IC_loss_mse(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_gru, y_true,pred_lgb=None):
        if pred_lgb is not None:
            mutualstacked =t.stack([pred_lgb, pred_gru], dim=0)
            mulualic = t.corrcoef(mutualstacked)[0][1]
            lamda = 0.05
            mseloss = t.mean((pred_gru - y_true) ** 2)
            return mseloss + lamda * mulualic
        else:
            return t.mean((pred_gru - y_true) ** 2)


class IC_loss_one(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_gru, y_true):
        predstacked = t.stack([y_true, pred_gru], dim=0)
        predIC = t.corrcoef(predstacked)[0][1]
        return - predIC


class multihead_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # Calculate the 10th percentile threshold
        # q_value = t.quantile(y_true, 0.1)
        q_value = 0

        # Create a binary mask for values below or equal to the 10th percentile
        mask = y_true >= q_value

        # Apply the mask to both predictions and targets
        filtered_y_pred = y_pred[mask]
        filtered_y_true = y_true[mask]

        # Compute the loss on the filtered values (e.g., MSE)
        if filtered_y_true.numel() > 0:  # Check if there are any elements
            loss = F.mse_loss(filtered_y_pred, filtered_y_true)
        else:
            loss = 0.0  # Handle case with no elements to avoid errors

        return loss


class mse_reg_mh(nn.Module):
    def __init__(self, alpha=0.17):
        super().__init__()
        self.mh = multihead_loss()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true) + self.alpha * self.mh(y_pred, y_true)
if __name__=='__main__':
    Gru_pred_df = pd.read_parquet("/home/hongkou/chenx/exp/exp_003/GRU_pred_df")

    xgb_pred = pd.read_parquet("/home/hongkou/chenx/exp/exp_003/xgb_pred")

    lable = pd.read_parquet("/home/hongkou/chenx/exp/exp_003/returns_df")
    lable = lable.fillna(0)
    corr0 = lable.T.corrwith(xgb_pred.T)
    corr = lable.T.corrwith(Gru_pred_df.T)
    corrwith = Gru_pred_df.T.corrwith(xgb_pred.T)
    ICloss = IC_loss_double_diff()
    for i in range(len(corr)):
        print(ICloss(t.tensor(xgb_pred.iloc[i,:]), t.tensor(Gru_pred_df.iloc[i,:]), t.tensor(lable.iloc[i,:])).numpy())
        if i == 20:
            break



















