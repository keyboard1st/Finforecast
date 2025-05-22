import numpy as np
import os
from utils.calculate_ic import ic_between_arr
import lightgbm as lgb
from get_data.xgb_dataloader import get_xgb_dataloader

def lgb_pred(device):
    '''
    :return: 两个arr
    '''
    model_path = r"/home/hongkou/chenx/exp/CY312lgbm.txt"
    xgb_model = lgb.Booster(params={'device': device},model_file=model_path)
    xgb_trainloader, xgb_testloader = get_xgb_dataloader("all", False)
    xgb_train_set = next(iter(xgb_trainloader))
    xgb_test_set = next(iter(xgb_testloader))
    def xgb_predict(xgb_model, dataloader):
        pred_list = []
        true_list = []
        for i, test in enumerate(dataloader):
            x, y = np.array(test[:, :-1]), np.array(test[:, -1])
            pred = xgb_model.predict(x)
            pred_list.append(pred)
            true_list.append(y)

        # 合并结果
        pred_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in pred_list], axis=0)
        true_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in true_list], axis=0)

        return pred_arr, true_arr
    xgb_test_pred, xgb_test_true = xgb_predict(xgb_model, xgb_test_set)     # (292, 3873)

    xgb_test_ic = ic_between_arr(xgb_test_pred, xgb_test_true)
    print("\nxgb test ic:", xgb_test_ic)
    return xgb_test_pred, xgb_test_true



if __name__=='__main__':
    xgb_test_pred, xgb_test_true = lgb_pred("cpu")

