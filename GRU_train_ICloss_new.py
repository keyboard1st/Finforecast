import random
import warnings

import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler

from metrics.calculate_ic import ic_between_arr
from metrics.log import *
from model.GRU_attention import AttGRU
from model.GRU_model import *
from model.TimeMixer import TimeMixer
from model.losses import get_criterion
from train.GRU_cross_time_train import norm_train, GRU_fin_test
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

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def GRU_train_and_test(config, TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader):
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
    trained_model, df_metrics = norm_train(config, TimeSeries_trainloader, TimeSeries_valiloader,
                                                               TimeSeries_testloader, model, early_stopping,
                                                               model_optim, criterion, scheduler, logger, scaler)

    last_model_path = os.path.join(exp_path, 'fin_last_model.pth')
    torch.save(trained_model.state_dict(), last_model_path)

    GRU_pred_arr, GRU_true = GRU_fin_test(TimeSeries_testloader, trained_model)
    GRU_ic = ic_between_arr(GRU_pred_arr, GRU_true)
    # IC统计结果
    logger.info(f"GRU final model test IC: {GRU_ic:.4f}")
    logger.info(f"GRU pred_arr shape:{GRU_pred_arr.shape}")

    ic_metrics = {
        "GRU Final IC": GRU_ic,
    }

    # 创建结构化DataFrame
    ic_df = pd.DataFrame.from_dict(
        ic_metrics,
        orient='index',
        columns=['IC Value']
    ).reset_index().rename(columns={'index': 'Metric'})

    result_df = ic_df.sort_values(by='IC Value', ascending=False)

    record_to_excel(df_config, df_metrics, result_df, exp_path, append=True)

if __name__ == '__main__':
    from get_data.minute_factors.min_CS_dataloader import get_min10_rollingtrain_TimeSeriesLoader
    # config.task_name = 'minute10_2022_2023'
    # config.time_period = '2022-2023'
    # config.device = 'cuda:5'
    # config.model_type = 'TimeMixer'
    # config.early_stop_patience = 3
    # config.loss = 'MSE'
    train_dataloader, val_dataloader, test_dataloader = get_min10_rollingtrain_TimeSeriesLoader(batchsize = 1, shuffle_time = True, window_size = 4, num_val_windows = 100, val_sample_mode = 'random', time_period = config.time_period, config = config)

    GRU_train_and_test(config, train_dataloader, val_dataloader, test_dataloader)

