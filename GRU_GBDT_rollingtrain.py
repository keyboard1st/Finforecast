import random
import warnings
from copy import deepcopy

import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler

from metrics.calculate_ic import ic_between_arr
from metrics.log import *
from model.GRU_attention import AttGRU
from model.GRU_model import *
from model.TimeMixer import TimeMixer
from model.losses import get_criterion
from train.GBDT_trainer import lgb_train_and_test, xgb_train_and_test, cat_train_and_test
from train.GRU_cross_time_train import train_and_cross_time_train, GRU_fin_test
from utils.tools import EarlyStopping

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from config import get_config
config = get_config()

random_seed = config.random_seed
random.seed(random_seed)
np.random.seed(random_seed)

def rollingtrain_GRU_and_GBDT(config, TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader, CrossSection_trainloader, CrossSection_testloader):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.exp_path = f'/home/hongkou/TimeSeries/exp/{config.task_name}'
    # 创建日志
    exp_path = config.exp_path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    if not os.path.exists(os.path.join(exp_path, 'models')):
        os.makedirs(os.path.join(exp_path, 'models'))
    logger = create_logger(os.path.join(exp_path))
    logger.info(f"Config loaded: {config.__dict__}")
    logger.info(f"Using device: {device}")
    df_config = pd.DataFrame([vars(config)])

    if config.model_type == 'GRU':
        model = GRU(config).to(device)
    elif config.model_type == 'BiGRU':
        model = BiGRU(config).to(device)
    elif config.model_type == 'two_GRU':
        model = two_GRU(config).to(device)
    elif config.model_type == 'AttGRU':
        model = AttGRU(config).to(device)
    elif config.model_type == 'TimeMixer':
        model = TimeMixer(config).to(device)
    else:
        raise ValueError("model_type error")
    logger.info(f"Using model: {model.model_name}")

    # 损失函数、优化器、学习率调度器、早停机制
    criterion = get_criterion(config.loss)
    logger.info(f"Using criterion: {criterion}")
    early_stopping = EarlyStopping(patience=config.early_stop_patience, verbose=True)

    if config.optimizer == 'Adam':
        model_optim = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'AdamW':
        model_optim = optim.AdamW(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'SGD':
        model_optim = optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError("optimizer error")
    logger.info(f'Using optimizer: {model_optim.__class__.__name__}')

    scaler = torch.amp.GradScaler(device="cuda")
    train_steps = len(TimeSeries_trainloader)
    if config.lradj == 'cos':
        warmup_epochs = 0  # 学习率上升阶段
        decay_epochs = int(config.train_epochs * 5 / 5)  # 学习率下降阶段
        scheduler = lr_scheduler.LambdaLR(optimizer=model_optim,
                                          lr_lambda=lambda epoch:
                                          # Warmup阶段：线性增长 (类似图像中property的逻辑分段)
                                          (epoch / warmup_epochs) if epoch < warmup_epochs else
                                          # Decay阶段：余弦下降 (比线性更平滑)
                                          (0.5 * (1 + torch.cos(torch.tensor(
                                              (epoch - warmup_epochs) / decay_epochs * torch.pi))))
                                          )
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=config.pct_start,
                                            epochs=config.train_epochs,
                                            max_lr=config.learning_rate)

    # 训练模型主函数
    trained_model, df_metrics = train_and_cross_time_train(config, TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader, model, early_stopping, model_optim, criterion, scheduler, logger, scaler)

    last_model_path = os.path.join(exp_path, 'fin_last_model.pth')
    torch.save(trained_model.state_dict(), last_model_path)

    GRU_pred_arr, GRU_true = GRU_fin_test(TimeSeries_testloader, trained_model)
    GRU_ic = ic_between_arr(GRU_pred_arr, GRU_true)
    # IC统计结果
    logger.info(f"GRU final model test IC: {GRU_ic:.4f}")
    logger.info(f"GRU pred_arr shape:{GRU_pred_arr.shape}")
    GRU_pred_df = pd.DataFrame(GRU_pred_arr)

    # 训练GBDT模型
    lgb_trainloader, lgb_testloader = deepcopy(CrossSection_trainloader), deepcopy(CrossSection_testloader)
    xgb_trainloader, xgb_testloader = deepcopy(CrossSection_trainloader), deepcopy(CrossSection_testloader)
    cat_trainloader, cat_testloader = CrossSection_trainloader, CrossSection_testloader
    lgb_pred_arr, lgb_ture = lgb_train_and_test(lgb_trainloader, lgb_testloader, config.exp_path)
    lgb_ic = ic_between_arr(lgb_pred_arr, lgb_ture)
    logger.info(f"LightGBM test IC: {lgb_ic:.4f}")
    xgb_pred_arr, xgb_true = xgb_train_and_test(xgb_trainloader, xgb_testloader, config.exp_path)
    xgb_ic = ic_between_arr(xgb_pred_arr, xgb_true)
    logger.info(f"xgBoost test IC: {xgb_ic:.4f}")
    cat_pred_arr, cat_true = cat_train_and_test(cat_trainloader, cat_testloader, config.exp_path)
    cat_ic = ic_between_arr(cat_pred_arr, cat_true)
    logger.info(f"catBoost test IC: {cat_ic:.4f}")

    assert xgb_pred_arr.shape == lgb_pred_arr.shape == cat_pred_arr.shape
    gbdt_pred_arr = (xgb_pred_arr + cat_pred_arr + lgb_pred_arr) / 3

    logger.info(f"gbdt_pred_arr shape:{gbdt_pred_arr.shape}")
    gbdt_pred_df = pd.DataFrame(gbdt_pred_arr)

    fin_factor_arr = (gbdt_pred_arr + GRU_pred_arr) / 2
    fin_factor = pd.DataFrame(fin_factor_arr)
    returns_df = pd.DataFrame(lgb_ture).reindex(index = fin_factor.index)

    corr = returns_df.T.corrwith(fin_factor.T)
    print("final concat ic：", corr.mean())

    mask3_corr = returns_df.iloc[3:,:].T.corrwith(fin_factor.iloc[3:,:].T)
    print("drop head 3 concat ic：", mask3_corr.mean())

    mask5_corr = returns_df.iloc[5:,:].T.corrwith(fin_factor.iloc[5:,:].T)
    print("drop head 5 concat ic：", mask5_corr.mean())

    mask15_corr = returns_df.iloc[15:,:].T.corrwith(fin_factor.iloc[15:,:].T)
    print("drop head 15 concat ic：", mask15_corr.mean())

    mask60_corr = returns_df.iloc[60:,:].T.corrwith(fin_factor.iloc[60:,:].T)
    print("drop head 60 concat ic：", mask60_corr.mean())

    ic_metrics = {
        "GRU Final IC": GRU_ic,
        "Final Concat IC": corr.mean(),
    }

    # 创建结构化DataFrame
    ic_df = pd.DataFrame.from_dict(
        ic_metrics,
        orient='index',
        columns=['IC Value']
    ).reset_index().rename(columns={'index': 'Metric'})

    result_df = ic_df.sort_values(by='IC Value', ascending=False)

    record_to_excel(df_config, df_metrics, result_df, exp_path, append=True)

    with_corr = gbdt_pred_df.T.corrwith(GRU_pred_df.T)
    print("with_corr：", with_corr.mean())


if __name__=='__main__':
    from get_data.DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingtrain_TimeSeriesLoader, get_DrJin129_rollingtrain_CrossSectionDatasetLoader
    # from get_data.CY312.CY312_rollingtrain_dataloader import get_CY312_rollingtrain_TimeSeriesLoader, get_CY312_rollingtrain_CrossSectionLoader
    # config.task_name = 'CY_2023_2024_twoGRU'
    # config.time_period = '2023-2024'
    # config.device = 'cuda:1'

    TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader = get_DrJin129_rollingtrain_TimeSeriesLoader(batchsize = 1, shuffle_time = config.shuffle_time, window_size = config.window_size,
                                                                                                                      num_val_windows = config.num_val_windows, val_sample_mode = 'random', time_period = config.time_period, config=config)
    CrossSection_trainloader, CrossSection_testloader = get_DrJin129_rollingtrain_CrossSectionDatasetLoader(batchsize="all", shuffle_time=False, time_period = config.time_period)
    rollingtrain_GRU_and_GBDT(config, TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader,CrossSection_trainloader, CrossSection_testloader)

    # config.task_name = 'Jin_2021_2022_test'
    # config.time_period = '2021-2022'
    # config.exp_path = f'/home/hongkou/chenx/exp/{config.task_name}'
    # rolling_train_exp(config, config.time_period)
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # config.task_name = 'Jin_2022_2023_test'
    # config.time_period = '2022-2023'
    # config.exp_path = f'/home/hongkou/chenx/exp/{config.task_name}'
    # rolling_train_exp(config, config.time_period)
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # config.task_name = 'Jin_2023_2024_test'
    # config.time_period = '2023-2024'
    # config.exp_path = f'/home/hongkou/chenx/exp/{config.task_name}'
    # rolling_train_exp(config, config.time_period)


    # setting = f'{config.task_name}_{config.model_type}_{config.input_dim}_hd{config.hidden_dim}_' \
    #                       f'nl{config.num_layers}_{config.output_dim}_lr{config.learning_rate}_' \
    #                       f'esp{config.early_stop_patience}_te{config.train_epochs}_' \
    #                       f'nvw{config.num_val_windows}_ps{config.pct_start}_la{config.lradj}'

