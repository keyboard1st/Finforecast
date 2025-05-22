import os
import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from metrics.backtest import *
from metrics.calculate_ic import models_ic_plot, plot_ic_bar
from model.GRU_model import *
from model.GRU_attention import AttGRU
from model.TimeMixer import TimeMixer


from train.GBDT_trainer import tree_test, tree_pred
from train.GRU_cross_time_train import GRU_fin_test, GRU_pred_market

# from get_data.DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingtrain_TimeSeriesLoader, get_DrJin129_rollingtrain_CrossSectionDatasetLoader
# from get_data.DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingfintest_TimeSeriesLoader, get_DrJin129_rollingfintest_CrossSectionDatasetLoader
from get_data.CY312.CY312_rollingtrain_dataloader import get_CY312_rollingtrain_TimeSeriesLoader, get_CY312_rollingtrain_CrossSectionLoader
from get_data.CY312.CY312_rollingtrain_dataloader import get_CY312_rollingfintest_TimeSeriesloader, get_CY312_rollingfintest_CrossSectionLoader

import lightgbm as lgb
import xgboost as xgb
import catboost as catb
from catboost import CatBoostRegressor, Pool

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from config import get_config


class path:
    # models:
    exp_path = '/home/hongkou/TimeSeries/exp/CY_2021_2022_std'
    time_period = '2021-2022'
    GRU_model_path = os.path.join(exp_path, 'models/last_model.pth')
    lgbm_path = os.path.join(exp_path, 'models/lgbm.txt')
    cat_path = os.path.join(exp_path, 'models/catb.cbm')
    xgb_path = os.path.join(exp_path, 'models/xgb.xgb')

    # index and columns source
    # market_align_outer_path = "/home/USB_DRIVE3/data_CX/chenx/data_warehouse/DrJin_factors_rolling/r_market_align_factor/2024-2025/all_F100_mkt_outer.parquet"
    market_align_outer_path = f'/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_market_align_factor/{time_period}/F100_mkt_outer.parquet'

    # for backtest
    # labels_out_path = '/home/USB_DRIVE3/data_CX/chenx/data_warehouse/DrJin_factors_rolling/r_label/2024-2025/label_outer.parquet'
    labels_out_path = f'/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_label/{time_period}/label_outer.parquet'

    # market_path = '/home/USB_DRIVE3/data_CX/chenx/data_warehouse/market_cap_2712_3883.parquet'
    market_path = '/home/hongkou/chenx/data_warehouse/marketcap.parquet'

    # save csv, plot...
    save_path = os.path.join(exp_path, 'pred_csv')
    plot_path = os.path.join(exp_path, 'plots')

config = get_config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
config.exp_path = path.exp_path
index_and_clos = pd.read_parquet(path.market_align_outer_path)
labels = pd.read_parquet(path.labels_out_path)
market_cap = pd.read_parquet(path.market_path)

if not os.path.exists(path.save_path):
    os.makedirs(path.save_path)
if not os.path.exists(path.plot_path):
    os.makedirs(path.plot_path)

if config.model_type == 'GRU':
    GRU_model = GRU(config)
elif config.model_type == 'BiGRU':
    GRU_model = BiGRU(config)
elif config.model_type == 'two_GRU':
    GRU_model = two_GRU(config)
elif config.model_type == 'AttGRU':
    GRU_model = AttGRU(config)
else:
    raise ValueError("model_type must be GRU or BiGRU")

# load model from path
state_dict = torch.load(path.GRU_model_path, map_location='cpu')
# 加载 state_dict
GRU_model.load_state_dict(state_dict)
# 然后将模型移动到 GPU（如果你需要）
GRU_model = GRU_model.to(device)
print(f'loading GRU model from{path.GRU_model_path}')

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(path.xgb_path)
lgb_model = lgb.Booster(model_file=path.lgbm_path)
cat_model = CatBoostRegressor().load_model(path.cat_path)

models = {
    "GRU": GRU_model,
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "CatBoost": cat_model
}

def test_all_model(TimeSeries_xy_loader, CrossSection_xy_loader, **models):
    model_pred_dict = {}  # 用来存各个模型的 pred_arr
    for name, model in models.items():
        if name == 'GRU':
            print('----------------------GRU test--------------------------')
            GRU_pred_arr, GRU_true = GRU_fin_test(TimeSeries_xy_loader, model)
            print("GRU pred shape ", GRU_pred_arr.shape)
            print("GRU true shape ", GRU_true.shape)
            model_pred_dict[name] = GRU_pred_arr
            model_pred_dict['label'] = GRU_true
        else:
            print(f'----------------------{name} test-----------------------')
            cur_loader = deepcopy(CrossSection_xy_loader)
            tree_pred_arr, tree_true = tree_test(cur_loader, model)
            print(f"{name} pred shape ", tree_pred_arr.shape)
            print(f"{name} true shape ", tree_true.shape)
            model_pred_dict[name] = tree_pred_arr

    model_pred_dict['LightGBM & CatBoost'] = (model_pred_dict['LightGBM'] + model_pred_dict['CatBoost']) / 2
    model_pred_dict['LightGBM & XGBoost & CatBoost'] = (model_pred_dict['XGBoost'] + model_pred_dict['LightGBM'] + model_pred_dict['CatBoost']) / 3
    model_pred_dict['GRU & LightGBM'] = (model_pred_dict['GRU'] + model_pred_dict['LightGBM']) / 2
    model_pred_dict['GRU & LightGBM & CatBoost'] = model_pred_dict['GRU'] / 2 + model_pred_dict['LightGBM'] / 4 + model_pred_dict['CatBoost'] / 4
    model_pred_dict['GRU & LightGBM & CatBoost & XgBoost'] = model_pred_dict['GRU'] / 2 + model_pred_dict['LightGBM'] / 6 + model_pred_dict['CatBoost'] / 6 + model_pred_dict['XGBoost'] / 6

    return model_pred_dict

def pred_all_model(TimeSeries_x_loader, CrossSection_x_loader, index_and_clos, **models):
    model_pred_df_dict = {}
    for name, model in models.items():
        if name == 'GRU':
            print('----------------------GRU Pred--------------------------')
            GRU_pred_arr = GRU_pred_market(TimeSeries_x_loader, model)
            print("GRU pred shape ", GRU_pred_arr.shape)
            GRU_fin_pred_df = pd.DataFrame(GRU_pred_arr)
            GRU_fin_pred_df.index = index_and_clos.index
            GRU_fin_pred_df.columns = index_and_clos.columns
            model_pred_df_dict[name] = GRU_fin_pred_df
        else:
            print(f'----------------------{name} Pred-----------------------')
            cur_loader = deepcopy(CrossSection_x_loader)
            tree_pred_arr = tree_pred(cur_loader, model)
            print(f"{name} pred shape ", tree_pred_arr.shape)
            tree_pred_df = pd.DataFrame(tree_pred_arr)
            tree_pred_df.index = index_and_clos.index
            tree_pred_df.columns = index_and_clos.columns
            model_pred_df_dict[name] = tree_pred_df

    model_pred_df_dict['LightGBM & CatBoost'] = (model_pred_df_dict['LightGBM'] + model_pred_df_dict['CatBoost']) / 2
    model_pred_df_dict['LightGBM & XGBoost & CatBoost'] = (model_pred_df_dict['XGBoost'] + model_pred_df_dict['LightGBM'] + model_pred_df_dict['CatBoost']) / 3
    model_pred_df_dict['GRU & LightGBM'] = (model_pred_df_dict['GRU'] + model_pred_df_dict['LightGBM']) / 2
    model_pred_df_dict['GRU & LightGBM & CatBoost'] = model_pred_df_dict['GRU'] / 2 + model_pred_df_dict['LightGBM'] / 4 + model_pred_df_dict['CatBoost'] / 4
    model_pred_df_dict['GRU & LightGBM & CatBoost & XgBoost'] = model_pred_df_dict['GRU'] / 2 + model_pred_df_dict['LightGBM'] / 6 + model_pred_df_dict['CatBoost'] / 6 + model_pred_df_dict['XGBoost'] / 6

    return model_pred_df_dict

def save_pred(model_pred_df_dict):
    ic_dict = {}
    for name, pred_df in model_pred_df_dict.items():
        pred_df_mask = pred_df.where(market_cap>0)
        pred_df_mask.to_csv(os.path.join(path.save_path, f'{name}_fin_pred_mask.csv'))
        pred_ic = (pred_df_mask.T.corrwith(labels.T)).mean()
        print(f'{name} Pred ic: {pred_ic}')
        ic_dict[name] = pred_ic
    return ic_dict


if __name__ == '__main__':
    _, _, GRU_label_align_xy_loader = get_CY312_rollingtrain_TimeSeriesLoader(time_period=path.time_period, config=config)
    _, tree_label_align_xy_loader = get_CY312_rollingtrain_CrossSectionLoader(time_period=path.time_period)
    GRU_mkt_align_x_loader = get_CY312_rollingfintest_TimeSeriesloader(time_period=path.time_period, config=config)
    trees_mkt_align_x_loader = get_CY312_rollingfintest_CrossSectionLoader(time_period=path.time_period)
    # test all models and caluculate TEST ic
    model_pred_dict = test_all_model(GRU_label_align_xy_loader, tree_label_align_xy_loader, **models)
    models_ic_plot(model_pred_dict, path.plot_path)
    model_pred_df_dict = pred_all_model(GRU_mkt_align_x_loader, trees_mkt_align_x_loader, index_and_clos, **models)
    ic_dict = save_pred(model_pred_df_dict)
    # 开始回测
    backtest_res = {}
    for name, pred_df in tqdm(model_pred_df_dict.items()):
        pred_df = pred_df.astype(np.float32)
        backtest_res[name] = backtest_strategy(pred_df, labels,market_cap)

    plot_model_metrics_and_save(backtest_res, path.plot_path)

    plot_ic_bar(ic_dict, path.plot_path)



