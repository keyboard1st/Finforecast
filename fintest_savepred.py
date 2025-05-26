import itertools
import warnings
from copy import deepcopy

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from metrics.backtest import *
from metrics.calculate_ic import ic_between_models_plot
from model.GRU_attention import AttGRU
from model.GRU_model import *
from model.TimeMixer import TimeMixer
from train.GBDT_trainer import tree_test, tree_pred
from train.GRU_cross_time_train import GRU_fin_test, GRU_pred_market

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from config import get_config


# ─── Paths & Config ───────────────────────────────────────────────────────────
class PathConfig:
    exp_path   = '/home/hongkou/TimeSeries/exp/CY_2021_2022'
    time_period = '2021-2022'
    save_path  = os.path.join(exp_path, 'pred_csv')
    plot_path  = os.path.join(exp_path, 'plots')

    # model weights
    GRU      = os.path.join(exp_path,   'models/best_model.pth')
    use_minute_model = 1
    TimeMixer   = os.path.join('/home/hongkou/TimeSeries/exp/minute10_2021_2022/models/best_model.pth')
    XGB      = os.path.join(exp_path,   'models/xgb.xgb')
    LGB      = os.path.join(exp_path,   'models/lgbm.txt')
    CAT      = os.path.join(exp_path,   'models/catb.cbm')

    # data sources
    market_align = f'/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_market_align_factor/{time_period}/F100_mkt_outer.parquet'
    labels       = f'/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_label/{time_period}/label_outer.parquet'
    market_cap   = '/home/hongkou/chenx/data_warehouse/marketcap.parquet'

# make sure dirs exist
os.makedirs(PathConfig.save_path, exist_ok=True)
os.makedirs(PathConfig.plot_path, exist_ok=True)

# ─── Load Config & Data ────────────────────────────────────────────────────────
config = get_config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
config.exp_path = PathConfig.exp_path

df_index_and_clos = pd.read_parquet(PathConfig.market_align)
labels_df         = pd.read_parquet(PathConfig.labels)
market_cap_df     = pd.read_parquet(PathConfig.market_cap)

# ─── Build & Load Models ───────────────────────────────────────────────────────

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

models = {
    "GRU": GRU_model,
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "CatBoost": cat_model,
    "TimeMixer": Minute_model
}

def test_all_model(TimeSeries_xy_loader, CrossSection_xy_loader, Minute_xy_loader = None, **models):
    model_pred_dict = {}  # 用来存各个模型的 pred_arr
    for name, model in models.items():
        if name == 'GRU':
            print('----------------------GRU test--------------------------')
            GRU_pred_arr, GRU_true = GRU_fin_test(TimeSeries_xy_loader, model)
            print("GRU pred shape ", GRU_pred_arr.shape)
            print("GRU true shape ", GRU_true.shape)
            model_pred_dict[name] = GRU_pred_arr
            model_pred_dict['label'] = GRU_true
        elif name == 'TimeMixer' and PathConfig.use_minute_model:
            print('----------------------TimeMixer test-----------------------')
            Minute_pred_arr, Minute_true = GRU_fin_test(Minute_xy_loader, model)
            print("Minute pred shape ", Minute_pred_arr.shape)
            print("Minute true shape ", Minute_true.shape)
            model_pred_dict[name] = Minute_pred_arr
        else:
            print(f'----------------------{name} test-----------------------')
            cur_loader = deepcopy(CrossSection_xy_loader)
            tree_pred_arr, tree_true = tree_test(cur_loader, model)
            print(f"{name} pred shape ", tree_pred_arr.shape)
            print(f"{name} true shape ", tree_true.shape)
            model_pred_dict[name] = tree_pred_arr

    return model_pred_dict

def pred_all_model(TimeSeries_x_loader, CrossSection_x_loader, index_and_clos, Minute_x_loader = None, **models):
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
        elif name == 'TimeMixer' and PathConfig.use_minute_model:
            print('----------------------TimeMixer Pred-----------------------')
            Minute_pred_arr = GRU_pred_market(Minute_x_loader, model)
            print("Minute pred shape ", Minute_pred_arr.shape)
            Minute_fin_pred_df = pd.DataFrame(Minute_pred_arr)
            Minute_fin_pred_df.index = index_and_clos.index
            Minute_fin_pred_df.columns = index_and_clos.columns
            model_pred_df_dict[name] = Minute_fin_pred_df
        else:
            print(f'----------------------{name} Pred-----------------------')
            cur_loader = deepcopy(CrossSection_x_loader)
            tree_pred_arr = tree_pred(cur_loader, model)
            print(f"{name} pred shape ", tree_pred_arr.shape)
            tree_pred_df = pd.DataFrame(tree_pred_arr)
            tree_pred_df.index = index_and_clos.index
            tree_pred_df.columns = index_and_clos.columns
            model_pred_df_dict[name] = tree_pred_df

    model_pred_df_dict['GBDT'] = (model_pred_df_dict['XGBoost'] + model_pred_df_dict['LightGBM'] + model_pred_df_dict['CatBoost']) / 3
    model_pred_df_dict.pop('XGBoost',None)
    model_pred_df_dict.pop('LightGBM',None)
    model_pred_df_dict.pop('CatBoost',None)

    return model_pred_df_dict

def get_weights():
    '''
    返回36种可能的权重：[[0.1, 0.1, 0.8], [0.1, 0.2, 0.7], [0.1, 0.3, 0.6], [0.1, 0.4, 0.5], [0.1, 0.5, 0.4], [0.1, 0.6, 0.3], [0.1, 0.7, 0.2], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.2, 0.2, 0.6]...]
    '''
    steps = [i / 10 for i in range(1, 10)]
    weights = []
    for w in itertools.product(steps, repeat=3):  # 三维
        if abs(sum(w) - 1.0) < 1e-8:  # 浮点数比较
            weights.append(list(w))
    return weights

def model_mixer(model_pred_df_dict):
    print('----------------------model mixer-----------------------')
    print('base models:', model_pred_df_dict.keys())
    avg_pred = (model_pred_df_dict['GRU'] + model_pred_df_dict['TimeMixer'] + model_pred_df_dict['GBDT']) / 3.0
    model_pred_df_dict['GRU/3 & TimeMixer/3 & GBDT/3'] = avg_pred

    weights = get_weights()  # 返回形如 [[0.1,0.1,0.8], ...] 的 36 个三元组
    ic_records = []
    for w1, w2, w3 in weights:
        # 加权预测
        mix_pred = (
                w1 * model_pred_df_dict['GRU'] +
                w2 * model_pred_df_dict['TimeMixer'] +
                w3 * model_pred_df_dict['GBDT']
        )
        # 只在市值>0的位置计算 IC
        masked = mix_pred.where(market_cap_df > 0)
        # 按列（股票）计算与 labels 的相关，再取平均
        ic = (masked.T.corrwith(labels_df.T)).mean()
        ic_records.append(((w1, w2, w3), ic, mix_pred))
        print(f'{w1}GRU & {w2}TimeMixer & {w3}GBDT, IC = {ic:.4f}')

    top5 = sorted(ic_records, key=lambda x: x[1], reverse=True)[:5]
    for idx, ((w1, w2, w3), ic, pred_df) in enumerate(top5, 1):
        name = f'{w1}GRU & {w2}TimeMixer & {w3}GBDT'
        print(f'top5 {name}, IC = {ic:.4f}')
        model_pred_df_dict[name] = pred_df
    print('final models:', list(model_pred_df_dict.keys()))
    return model_pred_df_dict

def save_pred(model_pred_df_dict):
    model_pred_df_dict_with_ic = {}
    for name, pred_df in model_pred_df_dict.items():
        pred_df_mask = pred_df.where(market_cap_df>0)
        pred_df_mask.to_csv(os.path.join(PathConfig.save_path, f'{name}_fin_pred_mask.csv'))
        pred_ic = (pred_df_mask.T.corrwith(labels_df.T)).mean()
        print(f'{name} Pred ic: {pred_ic:.4f}')
        new_name = name + str(format(pred_ic, '.4f'))
        model_pred_df_dict_with_ic[new_name] = pred_df_mask
    return model_pred_df_dict_with_ic


if __name__ == '__main__':
    # from get_data.CY312.CY312_rollingtrain_dataloader import get_CY312_rollingfintest_TimeSeriesloader, get_CY312_rollingfintest_CrossSectionLoader
    # from get_data.CY312.CY312_rollingtrain_dataloader import get_CY312_rollingtrain_TimeSeriesLoader, get_CY312_rollingtrain_CrossSectionLoader
    from get_data.DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingtrain_TimeSeriesLoader, get_DrJin129_rollingtrain_CrossSectionDatasetLoader
    from get_data.DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingfintest_TimeSeriesLoader, get_DrJin129_rollingfintest_CrossSectionDatasetLoader
    from get_data.minute_factors.min_CS_dataloader import get_min10_rollingtrain_TimeSeriesLoader, get_min10_rollingfintest_TimeSeriesloader

    _, _, GRU_label_align_xy_loader = get_DrJin129_rollingtrain_TimeSeriesLoader(time_period=PathConfig.time_period, config=config)
    _, _, Minute_label_align_xy_loader = get_min10_rollingtrain_TimeSeriesLoader(time_period=PathConfig.time_period, config=config)
    _, tree_label_align_xy_loader = get_DrJin129_rollingtrain_CrossSectionDatasetLoader(time_period=PathConfig.time_period)

    GRU_mkt_align_x_loader = get_DrJin129_rollingfintest_TimeSeriesLoader(time_period=PathConfig.time_period, config=config)
    Minute_mkt_align_x_loader = get_min10_rollingfintest_TimeSeriesloader(time_period=PathConfig.time_period, config=config)
    trees_mkt_align_x_loader = get_DrJin129_rollingfintest_CrossSectionDatasetLoader(time_period=PathConfig.time_period)

    # test all models and caluculate TEST ic
    model_pred_dict = test_all_model(GRU_label_align_xy_loader, tree_label_align_xy_loader, Minute_mkt_align_x_loader, **models)
    ic_between_models_plot(model_pred_dict, PathConfig.plot_path)
    model_pred_df_dict = pred_all_model(GRU_mkt_align_x_loader, trees_mkt_align_x_loader, df_index_and_clos, **models)
    model_pred_df_dict_with_ic = save_pred(model_pred_df_dict)

    # 开始回测
    backtest_res = {}
    for name, pred_df in tqdm(model_pred_df_dict_with_ic.items()):
        pred_df = pred_df.astype(np.float32)
        backtest_res[name] = backtest_strategy(pred_df, labels_df, market_cap_df)

    plot_model_metrics_and_save(backtest_res, PathConfig.plot_path)

    # plot_ic_bar(ic_dict, path.plot_path)



