import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST' or args.lradj == 'plateau' or args.lradj == 'cos':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def save_to_csv(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    data = pd.DataFrame({'true': true, 'preds': preds})
    data.to_csv(name, index=False, sep=',')


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_weights(weights, name='./pic/test.pdf'):
    """
    Weights visualization
    """
    fig, ax = plt.subplots()
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap='YlGnBu')
    fig.colorbar(im, pad=0.03, location='top')
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def create_model(config):
    """根据配置创建模型"""
    from model.GRU_model import GRU, BiGRU, two_GRU
    from model.GRU_attention import AttGRU
    from model.TimeMixer import TimeMixer
    from model.Encoder import TimeSeriesEncoder
    if config.model_type == 'GRU':
        return GRU(config)
    elif config.model_type == 'BiGRU':
        return BiGRU(config)
    elif config.model_type == 'two_GRU':
        return two_GRU(config)
    elif config.model_type == 'AttGRU':
        return AttGRU(config)
    elif config.model_type == 'Encoder':
        return TimeSeriesEncoder(config)
    elif config.model_type == 'TimeMixer':
        return TimeMixer(config)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")


def calculate_ensemble_prediction(gbdt_results):
    """计算GBDT集成预测"""
    xgb_pred_arr = gbdt_results['xgb_pred']
    lgb_pred_arr = gbdt_results['lgb_pred']
    cat_pred_arr = gbdt_results['cat_pred']
    # 验证形状一致性
    assert xgb_pred_arr.shape == lgb_pred_arr.shape == cat_pred_arr.shape, "GBDT模型预测形状不一致"
    # 简单平均集成
    return (xgb_pred_arr + cat_pred_arr + lgb_pred_arr) / 3
