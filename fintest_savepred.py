import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import warnings
import sys
if any('ipykernel_launcher' in a for a in sys.argv):
    sys.argv = sys.argv[:1]

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from metrics.backtest import BacktestEngine, backtest_strategy, plot_model_metrics_and_save
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
config.task_name = 'CCB_2019_2025_AttGRU_83'
config.exp_path = f'D:/chenxing/Finforecast/exp/{config.task_name}'
print('fin pred exp path',config.exp_path)
class PathConfig:
    exp_path   = config.exp_path
    time_period = config.test_time_period
    start_date = int(time_period.split('-')[0])
    end_date = int(time_period.split('-')[1])
    save_path  = os.path.join(exp_path, 'pred_csv')
    plot_path  = os.path.join(exp_path, 'plots')

    # model weights
    GRU      = os.path.join(exp_path,   'models/best_model.pth')
    use_minute_model = False
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
    elif config.factor_name == 'CCB':
        factor_outer_list = [f'D:/chenxing/Finforecast/factor_warehouse/factor_aligned/r_factor/{i}-{i + 1}/F1.parquet' for i in range(start_date, end_date)]
        labels_list = [f'D:/chenxing/Finforecast/factor_warehouse/factor_aligned/r_label/{i}-{i + 1}/label.parquet' for i in range(start_date, end_date)]

# make sure dirs exist
os.makedirs(PathConfig.save_path, exist_ok=True)
os.makedirs(PathConfig.plot_path, exist_ok=True)

# ─── Load Config & Data ────────────────────────────────────────────────────────
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

if config.factor_name == 'CCB':
    df_index_and_clos = pd.concat([pd.read_parquet(i) for i in PathConfig.factor_outer_list],axis=0)
    labels_df         = pd.concat([pd.read_parquet(i) for i in PathConfig.labels_list],axis=0)
    market_cap_df     = None
else:
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
    elif config.factor_name == 'CCB':
        from get_data.CCB.CCB_TimeSeries_dataloader import get_CCB_TimeSeriesDataloader, get_CCB_mkt_TimeSeriesDataloader
        from get_data.CCB.CCB_CrossSection_dataloader import get_CCB_CrossSectionDataloader, get_CCB_MKT_CrossSectionDataloader
        _, _, GRU_label_align_xy_loader = get_CCB_TimeSeriesDataloader(test_time_period=PathConfig.time_period,config=config)
        _, tree_label_align_xy_loader = get_CCB_CrossSectionDataloader(test_time_period=PathConfig.time_period)
        GRU_mkt_align_x_loader = get_CCB_mkt_TimeSeriesDataloader(test_time_period=PathConfig.time_period,config=config)
        trees_mkt_align_x_loader = get_CCB_MKT_CrossSectionDataloader(test_time_period=PathConfig.time_period)
    else:
        raise ValueError("Unsupported factor_name")
    

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
    model_pred_df_dict_mixed = model_mixer(PathConfig, model_pred_df_dict, labels_df, market_cap_df)

    # mixed model pred save
    model_pred_df_dict_with_ic = save_pred(PathConfig, model_pred_df_dict_mixed, labels_df, market_cap_df)

    # Enhanced backtest using BacktestEngine with TOP 10% strategy
    print("\n" + "="*80)
    print("Enhanced Backtest Analysis - TOP 10% Strategy")
    print("="*80)
    
    # Initialize backtest engine
    engine = BacktestEngine()
    
    # Convert data types for compatibility
    labels_df.index = labels_df.index.astype(np.int32)
    labels_df.columns = labels_df.columns.astype(np.int32)
    
    # Configure TOP 10% strategy
    strategy_config = {
        'name': 'TOP 10% Strategy',
        'top_pct': 0.1,
        'weight_method': 'equal'
    }
    
    # Run backtest for each model with TOP 10% strategy
    model_results = {}
    
    for model_name, pred_df in tqdm(model_pred_df_dict_with_ic.items(), desc="Running backtests"):
        try:
            # Ensure data types match
            pred_df = pred_df.astype(np.float32)
            pred_df.index = pred_df.index.astype(np.int32)
            pred_df.columns = pred_df.columns.astype(np.int32)
            
            # Run single strategy backtest
            weights, results = engine.backtest_strategy(
                pred_df, 
                labels_df, 
                market_cap=market_cap_df,
                top_pct=strategy_config['top_pct'],
                weight_method=strategy_config['weight_method']
            )
            
            # Store results
            model_results[model_name] = {
                'weights': weights,
                'metrics': results
            }
            
            # Print summary for this model
            print(f"\n{model_name} Results:")
            print(f"  Annualized Return: {results['annualized_return']:8.2%}")
            print(f"  Sharpe Ratio:      {results['sharpe_ratio']:8.3f}")
            print(f"  Max Drawdown:      {results['max_drawdown']:8.2%}")
            print(f"  Total Return:      {results['total_return']:8.2%}")
            print(f"  Volatility:        {results['volatility']:8.2%}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    # Create enhanced plots with group analysis
    if model_results:
        print(f"\nGenerating enhanced analysis plots...")
        
        # Select the best model for detailed group analysis (by Sharpe ratio)
        best_model = max(model_results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
        best_model_name, best_model_data = best_model
        
        print(f"Best performing model: {best_model_name} (Sharpe: {best_model_data['metrics']['sharpe_ratio']:.3f})")
        
        # Use the best model's predictions for group analysis
        best_pred_df = model_pred_df_dict_with_ic[best_model_name]
        best_pred_df = best_pred_df.astype(np.float32)
        best_pred_df.index = best_pred_df.index.astype(np.int32)
        best_pred_df.columns = best_pred_df.columns.astype(np.int32)
        
        # Create enhanced plots for all models
        engine.create_enhanced_plots(
            model_results, 
            save_path=PathConfig.plot_path,
            pred_returns=best_pred_df,  # Use best model for group analysis
            labels=labels_df
        )
        
        print(f"Enhanced plots saved to: {PathConfig.plot_path}")
        
        # Summary table
        print(f"\n" + "="*80)
        print("Model Performance Summary (TOP 10% Strategy)")
        print("="*80)
        print(f"{'Model':<20} {'Ann.Return':<12} {'Sharpe':<8} {'Max DD':<8} {'Volatility':<12}")
        print("-"*80)
        
        for name, result in model_results.items():
            metrics = result['metrics']
            print(f"{name:<20} {metrics['annualized_return']:>10.2%} "
                  f"{metrics['sharpe_ratio']:>7.3f} {metrics['max_drawdown']:>7.2%} "
                  f"{metrics['volatility']:>10.2%}")
        
        print("\nNote: Group analysis (decile/quintile) shows the model's stock ranking ability.")
        print("It's independent of the TOP 10% selection strategy.")
        
    else:
        print("No successful backtest results to plot.")




