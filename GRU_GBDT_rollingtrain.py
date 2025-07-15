import random
import warnings
from copy import deepcopy

import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from metrics.calculate_ic import ic_between_arr, calculate_ic_metrics
from metrics.log import save_experiment_results, create_logger
from model.losses import get_criterion
from train.GBDT_trainer import train_gbdt_ensemble_models
from train.trainer import create_trainer, ModelTrainer
from utils.tools import EarlyStopping, create_model, calculate_ensemble_prediction

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from config import get_config
config = get_config()

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


def rollingtrain_GRU_and_GBDT(config, TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader, CrossSection_trainloader, CrossSection_testloader):
    """
    滚动训练GRU和GBDT模型的主函数
    """
    # 设置实验路径
    config.exp_path = f'D:\\chenxing\\Finforecast\\exp\\{config.task_name}'
    exp_path = config.exp_path
    
    # 创建必要的目录
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'models'), exist_ok=True)
    
    # 创建日志记录器
    logger = create_logger(os.path.join(exp_path))
    logger.info(f"Config loaded: {config.__dict__}")
    logger.info(f"Using device: {config.device}")
    
    # 保存配置到DataFrame
    df_config = pd.DataFrame([vars(config)])

    # 创建模型
    model = create_model(config)
    logger.info(f"Using model: {model.__class__.__name__}")

    # 创建训练器
    trainer = create_trainer(config, model, config.device)
    
    # 训练模型
    logger.info("Start training time series model...")
    trained_model, df_metrics = trainer.train(
        TimeSeries_trainloader, 
        TimeSeries_valiloader, 
        TimeSeries_testloader, 
        logger
    )

    # 保存最终模型
    last_model_path = os.path.join(exp_path, 'fin_last_model.pth')
    torch.save(trained_model.state_dict(), last_model_path)

    # 测试模型性能
    logger.info("Start model testing...")
    GRU_pred_arr, GRU_true = trainer.test(TimeSeries_testloader)
    GRU_ic = ic_between_arr(GRU_pred_arr, GRU_true)
    
    logger.info(f"GRU final model test IC: {GRU_ic:.4f}")
    logger.info(f"GRU prediction array shape: {GRU_pred_arr.shape}")
    GRU_pred_df = pd.DataFrame(GRU_pred_arr)

    # 训练GBDT模型
    logger.info("Start training GBDT models...")
    gbdt_results, lgb_model, xgb_model, cat_model = train_gbdt_ensemble_models(CrossSection_trainloader, CrossSection_testloader, config.exp_path, logger)
    
    # 计算集成预测
    gbdt_pred_arr = calculate_ensemble_prediction(gbdt_results)
    logger.info(f"GBDT prediction array shape: {gbdt_pred_arr.shape}")
    gbdt_pred_df = pd.DataFrame(gbdt_pred_arr)

    # 计算最终因子
    fin_factor_arr = (gbdt_pred_arr + GRU_pred_arr) / 2
    fin_factor = pd.DataFrame(fin_factor_arr)
    returns_df = pd.DataFrame(gbdt_results['lgb_true']).reindex(index=fin_factor.index)

    # 计算各种IC指标
    ic_metrics = calculate_ic_metrics(returns_df, fin_factor, logger)
    
    # 计算相关性
    with_corr = gbdt_pred_df.T.corrwith(GRU_pred_df.T)
    logger.info(f"GBDT and GRU correlation: {with_corr.mean():.4f}")

    # 保存结果
    save_experiment_results(df_config, df_metrics, ic_metrics, exp_path, logger)

    return trained_model, lgb_model, xgb_model, cat_model


if __name__=='__main__':
    from get_data.DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingtrain_TimeSeriesLoader, get_DrJin129_rollingtrain_CrossSectionDatasetLoader
    from get_data.CY312.CY312_rollingtrain_dataloader import get_CY312_rollingtrain_TimeSeriesLoader, get_CY312_rollingtrain_CrossSectionLoader
    from get_data.CCB.CCB_dataloader import get_CCB_TimeSeriesDataloader, get_CCB_CrossSectionDataloader

    # # 设置配置参数
    config.task_name = 'Encoder_hd256_133_202401_202504'
    config.train_time_period = '201901-202312'
    config.test_time_period = '202401-202504'

    # 根据因子类型加载数据
    if config.factor_name == 'CY312':
        TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader = get_CY312_rollingtrain_TimeSeriesLoader(
            batchsize=1, shuffle_time=config.shuffle_time, window_size=config.window_size,
            num_val_windows=config.num_val_windows, val_sample_mode='random', 
            time_period=config.time_period, config=config
        )
        CrossSection_trainloader, CrossSection_testloader = get_CY312_rollingtrain_CrossSectionLoader(
            batchsize="all", shuffle_time=False, time_period=config.time_period
        )
    elif config.factor_name == 'DrJin129':
        TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader = get_DrJin129_rollingtrain_TimeSeriesLoader(
            batchsize=1, shuffle_time=config.shuffle_time, window_size=config.window_size,
            num_val_windows=config.num_val_windows, val_sample_mode='random', 
            time_period=config.time_period, config=config
        )
        CrossSection_trainloader, CrossSection_testloader = get_DrJin129_rollingtrain_CrossSectionDatasetLoader(
            batchsize="all", shuffle_time=False, time_period=config.time_period
        )
    elif config.factor_name == 'CCB':
        TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader = get_CCB_TimeSeriesDataloader(
            shuffle_time=config.shuffle_time, train_time_period=config.train_time_period, val_sample_mode='random_days',
            test_time_period=config.test_time_period, config=config
        )
        CrossSection_trainloader, CrossSection_testloader = get_CCB_CrossSectionDataloader(
            train_time_period=config.train_time_period, test_time_period=config.test_time_period
        )
    else:
        raise ValueError(f"Unsupported factor type: {config.factor_name}")

    # 开始训练
    rollingtrain_GRU_and_GBDT(config, TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader, CrossSection_trainloader, CrossSection_testloader)


