
import warnings

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from metrics.backtest import *
from metrics.calculate_ic import ic_between_models_plot
from metrics.models_pred import *
from model.GRU_attention import AttGRU
from model.GRU_model import *
from model.TimeMixer import TimeMixer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from config import get_config


# ─── Paths & Config ───────────────────────────────────────────────────────────
config = get_config()
config.task_name = 'Jin_2021_2022_randomseed2025'
config.exp_path = f'/home/hongkou/TimeSeries/exp/{config.task_name}'
print('fin pred exp path',config.exp_path)
class PathConfig:
    exp_path   = config.exp_path
    time_period = config.time_period
    save_path  = os.path.join(exp_path, 'pred_csv')
    plot_path  = os.path.join(exp_path, 'plots')

    # model weights
    GRU      = os.path.join(exp_path,   'models/best_model.pth')
    use_minute_model = True
    TimeMixer   = os.path.join('/home/hongkou/TimeSeries/exp/minute10_2021_2022/models/best_model.pth')
    XGB      = os.path.join(exp_path,   'models/xgb.xgb')
    LGB      = os.path.join(exp_path,   'models/lgbm.txt')
    CAT      = os.path.join(exp_path,   'models/catb.cbm')

    # data sources
    if config.factor_name == 'CY312':
        market_align = f'/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_market_align_factor/{time_period}/F100_mkt_outer.parquet'
        labels       = f'/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_label/{time_period}/label_outer.parquet'
        market_cap   = '/home/hongkou/chenx/data_warehouse/marketcap.parquet'
    elif config.factor_name == 'DrJin129':
        market_align = f'/home/USB_DRIVE1/Chenx_datawarehouse/DrJin/factors_rolling/r_market_align_factor/{time_period}/all_F100_mkt_outer.parquet'
        labels       = f'/home/USB_DRIVE1/Chenx_datawarehouse/DrJin/factors_rolling/r_label/{time_period}/label_outer.parquet'
        market_cap   = '/home/hongkou/chenx/data_warehouse/market_cap_2712_3883.parquet'

# make sure dirs exist
os.makedirs(PathConfig.save_path, exist_ok=True)
os.makedirs(PathConfig.plot_path, exist_ok=True)

# ─── Load Config & Data ────────────────────────────────────────────────────────
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

df_index_and_clos = pd.read_parquet(PathConfig.market_align)
labels_df         = pd.read_parquet(PathConfig.labels)
market_cap_df     = pd.read_parquet(PathConfig.market_cap)

# ─── Build & Load Models ───────────────────────────────────────────────────────
print('Loading model from', os.path.join(PathConfig.exp_path,'models'))

if config.model_type == 'GRU':
    GRU_model = GRU(config)
elif config.model_type == 'BiGRU':
    GRU_model = BiGRU(config)
elif config.model_type == 'two_GRU':
    GRU_model = two_GRU(config)
elif config.model_type == 'AttGRU':
    GRU_model = AttGRU(config)
else:
    raise ValueError("Unsupported model_type")

# 加载 GRU 模型
GRU_model.load_state_dict(torch.load(PathConfig.GRU, map_location='cpu'))
GRU_model = GRU_model.to(device)
print(f'Loaded GRU model from {PathConfig.GRU}')

if PathConfig.use_minute_model:
    # 加载 TimeMixer 模型
    Minute_model = TimeMixer(config)
    Minute_model.load_state_dict(torch.load(PathConfig.TimeMixer, map_location='cpu'))
    Minute_model = Minute_model.to(device)
    print(f'Loaded minute model from {PathConfig.TimeMixer}')

# 加载树模型
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(PathConfig.XGB)
lgb_model = lgb.Booster(model_file=PathConfig.LGB)
cat_model = CatBoostRegressor().load_model(PathConfig.CAT)

if PathConfig.use_minute_model:
    models = {
        "GRU": GRU_model,
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "CatBoost": cat_model,
        "TimeMixer": Minute_model
    }
else:
    models = {
        "GRU": GRU_model,
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "CatBoost": cat_model
    }


if __name__ == '__main__':
    # data load
    if config.factor_name == 'CY312':
        from get_data.CY312.CY312_rollingtrain_dataloader import get_CY312_rollingfintest_TimeSeriesloader, get_CY312_rollingfintest_CrossSectionLoader
        from get_data.CY312.CY312_rollingtrain_dataloader import get_CY312_rollingtrain_TimeSeriesLoader, get_CY312_rollingtrain_CrossSectionLoader
        _, _, GRU_label_align_xy_loader = get_CY312_rollingtrain_TimeSeriesLoader(time_period=PathConfig.time_period,config=config)
        _, tree_label_align_xy_loader = get_CY312_rollingtrain_CrossSectionLoader(time_period=PathConfig.time_period)
        GRU_mkt_align_x_loader = get_CY312_rollingfintest_TimeSeriesloader(time_period=PathConfig.time_period,config=config)
        trees_mkt_align_x_loader = get_CY312_rollingfintest_CrossSectionLoader(time_period=PathConfig.time_period)
    elif config.factor_name == 'DrJin129':
        from get_data.DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingtrain_TimeSeriesLoader, get_DrJin129_rollingtrain_CrossSectionDatasetLoader
        from get_data.DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingfintest_TimeSeriesLoader, get_DrJin129_rollingfintest_CrossSectionDatasetLoader
        _, _, GRU_label_align_xy_loader = get_DrJin129_rollingtrain_TimeSeriesLoader(time_period=PathConfig.time_period,config=config)
        _, tree_label_align_xy_loader = get_DrJin129_rollingtrain_CrossSectionDatasetLoader(time_period=PathConfig.time_period)
        GRU_mkt_align_x_loader = get_DrJin129_rollingfintest_TimeSeriesLoader(time_period=PathConfig.time_period,config=config)
        trees_mkt_align_x_loader = get_DrJin129_rollingfintest_CrossSectionDatasetLoader(time_period=PathConfig.time_period)

    # all model test and load
    if PathConfig.use_minute_model:
        from get_data.minute_factors.min_CS_dataloader import get_min10_rollingtrain_TimeSeriesLoader, get_min10_rollingfintest_TimeSeriesloader
        _, _, Minute_label_align_xy_loader = get_min10_rollingtrain_TimeSeriesLoader(time_period=PathConfig.time_period, config=config)
        Minute_mkt_align_x_loader = get_min10_rollingfintest_TimeSeriesloader(time_period=PathConfig.time_period, config=config)
        model_pred_dict = test_all_model(PathConfig, GRU_label_align_xy_loader, tree_label_align_xy_loader,Minute_label_align_xy_loader, **models)
        ic_between_models_plot(model_pred_dict, PathConfig.plot_path)
        model_pred_df_dict = pred_all_model(PathConfig, GRU_mkt_align_x_loader, trees_mkt_align_x_loader,df_index_and_clos, Minute_mkt_align_x_loader, **models)
    else:
        model_pred_dict = test_all_model(PathConfig, GRU_label_align_xy_loader, tree_label_align_xy_loader, **models)
        ic_between_models_plot(model_pred_dict, PathConfig.plot_path)
        model_pred_df_dict = pred_all_model(PathConfig, GRU_mkt_align_x_loader, trees_mkt_align_x_loader,df_index_and_clos, **models)

    # model mixer
    model_pred_df_dict_mixed = model_mixer(PathConfig, model_pred_df_dict, market_cap_df, labels_df)

    # mixed model pred save
    model_pred_df_dict_with_ic = save_pred(PathConfig, model_pred_df_dict_mixed, market_cap_df, labels_df)

    # backtest
    backtest_res = {}
    for name, pred_df in tqdm(model_pred_df_dict_with_ic.items()):
        pred_df = pred_df.astype(np.float32)
        backtest_res[name] = backtest_strategy(pred_df, labels_df, market_cap_df)

    plot_model_metrics_and_save(backtest_res, PathConfig.plot_path)




