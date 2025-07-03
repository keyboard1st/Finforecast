import torch
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import os
from copy import deepcopy
from metrics.calculate_ic import ic_between_arr_new
from metrics.calculate_ic import ic_between_arr

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from config import get_config
config = get_config()


def tree_test(dataloader, model):
    """
    树的测试方法
    不fillna
    """
    xgb_test_set = next(iter(dataloader))
    pred_list = []
    true_list = []
    for i, test in enumerate(xgb_test_set):
        x, y = np.array(test[:, :-1]), np.array(test[:, -1])
        pred = model.predict(x)
        pred_list.append(pred)
        true_list.append(y)
    # 合并结果
    pred_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in pred_list], axis=0)
    true_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in true_list], axis=0)

    test_ic = ic_between_arr_new(pred_arr, true_arr)
    print("\nlgbm test ic:", test_ic)

    return pred_arr, true_arr

def tree_pred(mkt_align_x_loader, model):
    '''
    接受market_align_dataloader
    '''
    xgb_test_set = next(iter(mkt_align_x_loader))
    pred_list = []
    for i, test in enumerate(xgb_test_set):
        x = np.array(test)
        pred = model.predict(x)
        pred_list.append(pred)
    pred_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in pred_list], axis=0)
    return pred_arr

def lgb_train_and_test(trainloader, testloader, exp_path):
    '''
    :return: 两个arr
    '''
    train_set = next(iter(trainloader))
    train_data = train_set.reshape(-1, train_set.shape[-1])
    mask = ~torch.isnan(train_data[:, -1])
    train_data = train_data[mask]

    # train_nan_mask = torch.isnan(train_data)
    # train_nan_counts = train_nan_mask.sum(dim=1)
    # train_indices = torch.where(train_nan_counts <= 8)[0]
    # train_data = train_data[train_indices]

    train_data = np.array(train_data)
    train_data_x = train_data[:, :-1]
    train_data_y = train_data[:, -1]

    model = lgb.LGBMRegressor(learning_rate=0.07, n_estimators=350)

    model.fit(train_data_x, train_data_y)
    model.booster_.save_model(os.path.join(exp_path, 'models/lgbm.txt'))
    print(f"lgbm save to {os.path.join(exp_path, 'models/lgbm.txt')}")

    test_pred, test_true = tree_test(testloader, model)     # (292, 3873)

    return test_pred, test_true

def xgb_train_and_test(trainloader, testloader, exp_path):
    '''
    :return: 两个arr
    '''

    train_set = next(iter(trainloader))
    train_data = train_set.reshape(-1, train_set.shape[-1])
    mask = ~torch.isnan(train_data[:, -1])
    train_data = train_data[mask]

    # train_nan_mask = torch.isnan(train_data)
    # train_nan_counts = train_nan_mask.sum(dim=1)
    # train_indices = torch.where(train_nan_counts <= 8)[0]
    # train_data = train_data[train_indices]

    train_data = np.array(train_data)
    train_data_x = train_data[:, :-1]
    train_data_y = train_data[:, -1]

    model = xgb.XGBRegressor(
        learning_rate=0.07,
        n_estimators=200,
        verbosity=0  # Equivalent to CatBoost's verbose=False
    )
    model.fit(
        X=train_data_x,
        y=train_data_y,
        # Optional: Add eval_set for early stopping
        # eval_set=Pool(valid_data_x, valid_data_y),
        # early_stopping_rounds=45
    )
    model.save_model(os.path.join(exp_path, 'models/xgb.xgb'))

    test_pred, test_true = tree_test(testloader, model)
    return test_pred, test_true

def cat_train_and_test(trainloader, testloader, exp_path):
    '''
    :return: 两个arr
    '''

    train_set = next(iter(trainloader))
    train_data = train_set.reshape(-1, train_set.shape[-1])
    mask = ~torch.isnan(train_data[:, -1])
    train_data = train_data[mask]

    # train_nan_mask = torch.isnan(train_data)
    # train_nan_counts = train_nan_mask.sum(dim=1)
    # train_indices = torch.where(train_nan_counts <= 8)[0]
    # train_data = train_data[train_indices]

    train_data = np.array(train_data)
    train_data_x = train_data[:, :-1]
    train_data_y = train_data[:, -1]

    model = CatBoostRegressor(eta=0.07, iterations=500, verbose=False)
    model.fit(
        X=train_data_x,
        y=train_data_y,
        # Optional: Add eval_set for early stopping
        # eval_set=Pool(valid_data_x, valid_data_y),
        # early_stopping_rounds=45
    )
    model.save_model(os.path.join(exp_path, 'models/catb.cbm'))


    test_pred, test_true = tree_test(testloader, model)
    return test_pred, test_true

def train_gbdt_ensemble_models(train_loader, test_loader, exp_path, logger=None):
    """
    训练所有GBDT模型并返回预测结果字典
    :return: dict 包含lgb、xgb、catboost的预测、真实值和IC
    """
    from copy import deepcopy
    from metrics.calculate_ic import ic_between_arr
    # 创建数据加载器的副本以避免数据冲突
    lgb_trainloader, lgb_testloader = deepcopy(train_loader), deepcopy(test_loader)
    xgb_trainloader, xgb_testloader = deepcopy(train_loader), deepcopy(test_loader)
    cat_trainloader, cat_testloader = train_loader, test_loader
    
    # 训练LightGBM
    if logger: logger.info("训练LightGBM模型...")
    lgb_pred_arr, lgb_true = lgb_train_and_test(lgb_trainloader, lgb_testloader, exp_path)
    lgb_ic = ic_between_arr(lgb_pred_arr, lgb_true)
    if logger: logger.info(f"LightGBM测试IC: {lgb_ic:.4f}")
    
    # 训练XGBoost
    if logger: logger.info("训练XGBoost模型...")
    xgb_pred_arr, xgb_true = xgb_train_and_test(xgb_trainloader, xgb_testloader, exp_path)
    xgb_ic = ic_between_arr(xgb_pred_arr, xgb_true)
    if logger: logger.info(f"XGBoost测试IC: {xgb_ic:.4f}")
    
    # 训练CatBoost
    if logger: logger.info("训练CatBoost模型...")
    cat_pred_arr, cat_true = cat_train_and_test(cat_trainloader, cat_testloader, exp_path)
    cat_ic = ic_between_arr(cat_pred_arr, cat_true)
    if logger: logger.info(f"CatBoost测试IC: {cat_ic:.4f}")
    
    return {
        'lgb_pred': lgb_pred_arr,
        'lgb_true': lgb_true,
        'lgb_ic': lgb_ic,
        'xgb_pred': xgb_pred_arr,
        'xgb_true': xgb_true,
        'xgb_ic': xgb_ic,
        'cat_pred': cat_pred_arr,
        'cat_true': cat_true,
        'cat_ic': cat_ic
    }

if __name__=='__main__':
    pass



