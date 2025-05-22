import torch.optim as optim
from torch.optim import lr_scheduler

from model.losses import get_criterion
from metrics.log import *
from metrics.calculate_ic import ic_between_arr, ic_between_arr_new
from utils.tools import EarlyStopping
from train.GRU_cross_time_train import train_and_cross_time_train, GRU_fin_test

from model.GRU_model import *
from model.TimeMixer import TimeMixer
from model.GRU_attention import AttGRU
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from config import get_config
config = get_config()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


setting = f'{config.model_type}_{config.input_dim}_hd{config.hidden_dim}_' \
                      f'nl{config.num_layers}_dr{config.dropout}_{config.output_dim}'
config.exp_path = f'/home/hongkou/chenx/exp/{config.task_name}'

def GRU_train_and_test(config, setting, train_dataloader, val_dataloader, test_dataloader):
    # 创建日志
    exp_path = config.exp_path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    logger = create_logger(os.path.join(exp_path))
    logger.info(f"Config loaded: {config.__dict__}")
    df_config = pd.DataFrame([vars(config)])

    # 加载模型
    lgb_model = lgb.Booster(params={'device': device}, model_file=os.path.join(config.exp_path, 'lgbm.txt')) if config.cross_train else None
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
    model_optim = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(device="cuda")
    train_steps = len(train_dataloader)
    if config.lradj == 'cos':
        warmup_epochs = 0 # 学习率上升阶段
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
    trained_model, df_metrics = train_and_cross_time_train(train_dataloader, val_dataloader, test_dataloader, model, early_stopping, model_optim, criterion, scheduler, logger, scaler)

    fin_model_path = os.path.join(exp_path, f'{setting}.pth')
    torch.save(trained_model.state_dict(), fin_model_path)

    GRU_pred_arr, GRU_true = GRU_fin_test(test_dataloader, trained_model)
    GRU_ic = ic_between_arr(GRU_pred_arr, GRU_true)
    # IC统计结果
    logger.info(f"GRU final model test IC: {GRU_ic:.4f}")
    print(f"GRU final model test IC: {GRU_ic:.4f}")
    print("GRU_pred_arr shape:", GRU_pred_arr.shape)
    GRU_pred_df = pd.DataFrame(GRU_pred_arr)
    GRU_pred_df.to_parquet(os.path.join(exp_path, "GRU_pred"))

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

    train_dataloader, val_dataloader, test_dataloader = get_CSDay_dataloader(batchsize=1,
                                                                             shuffle_time=config.shuffle_time,
                                                                             window_size=config.window_size,
                                                                             num_val_windows=config.num_val_windows,
                                                                             val_sample_mode='random')

    GRU_train_and_test(config, setting, train_dataloader, val_dataloader, test_dataloader)














