import argparse
import os

def get_config():
    """使用argparse配置实验参数"""
    parser = argparse.ArgumentParser(description="实验参数配置")

    # 训练任务配置
    parser.add_argument('--task_name', type=str, default='CY_2021_2022', help='实验任务')
    parser.add_argument('--train_model', type=str, default='rollingtrain',choices=['rollingtrain', 'last_year_train'], help='训练模式')
    parser.add_argument('--time_period', type=str, default='2021-2022', help='用于滚动训练的时间段标识')
    parser.add_argument('--factor_name', type=str, default='CY312',choices=['Ding128', 'CY312', 'DrJin129'], help='因子类型')
    parser.add_argument('--device', type=str, default='cuda:1',help='运行设备')

    # 模型基础配置
    parser.add_argument('--model_type', type=str, default='AttGRU', choices=['GRU', 'BiGRU', 'two_GRU', 'AttGRU','TimeMixer'], help='模型类型')
    parser.add_argument('--input_dim', type=int, default=312, help='输入特征维度')
    parser.add_argument('--window_size', type=int, default=30, help='滑动窗口大小')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='网络层数')
    parser.add_argument('--output_dim', type=int, default=1, help='输出维度')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout概率')
    parser.add_argument('--num_heads', type=int, default=4, help='Attention头数')

    # 训练配置
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD'], help='优化器选择')
    parser.add_argument('--shuffle_time', default=True, help='是否打乱时间序列')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='早停等待轮数')
    parser.add_argument('--train_epochs', type=int, default=50, help='训练轮数')

    # 数据配置
    parser.add_argument('--num_val_windows', type=int, default=100, help='验证集窗口数量')
    parser.add_argument('--val_sample_mode', type=str, default="random",choices=["random", "tail"], help='验证集采样模式')

    # 损失函数
    parser.add_argument('--loss', type=str, default='MHMSE',choices=['MSE', 'MAE', 'Huber', 'MHMSE'], help='损失函数类型')

    # 学习率调度
    parser.add_argument('--pct_start', type=float, default=0.2, help='学习率预热比例')
    parser.add_argument('--lradj', type=str, default='TST',choices=['TST', 'cos'], help='学习率调整策略')

    # TimeMixer配置
    parser.add_argument('--seq_len', type=int, default=24,help='输入序列长度')
    parser.add_argument('--enc_in', type=int, default=129,help='Encoder 输入通道数（等同 input_dim）')
    parser.add_argument('--dec_in', type=int, default=129,help='Decoder 输入通道数（等同 input_dim）')
    parser.add_argument('--c_out', type=int, default=129,help='模型输出通道数（等同 input_dim）')
    parser.add_argument('--down_sampling_layers', type=int, default=2,help='下采样层数')
    parser.add_argument('--down_sampling_window', type=int, default=2,help='下采样窗口大小')
    parser.add_argument('--e_layers', type=int, default=3,help='PBM 层数')
    parser.add_argument('--down_sampling_method', type=str, default='avg',help='下采样方法：max, avg, conv')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',help='分解趋势方法：moving_avg or dft_decomp')
    parser.add_argument('--moving_avg', type=int, default=25,help='移动平均窗口大小')
    parser.add_argument('--pred_len', type=int, default=1,help='预测步长')
    parser.add_argument('--d_ff', type=int, default=32,help='中间隐藏层维度')
    parser.add_argument('--use_norm', type=int, default=0,help='是否使用归一化 (0: 关, 1: 开)')

    # 路径配置
    parser.add_argument('--exp_path', type=str, default='/home/hongkou/TimeSeries/exp/',
                        help='实验输出路径')

    # 交叉验证
    parser.add_argument('--cross_train', default=False, help='是否使用窗口交叉验证')

    # 数据存储地址
    parser.add_argument("--Ding128_alb_inner_path",
        type=str,
        default=r"/home/USB_DRIVE3/data_CX/chenx/data_warehouse/Ding_factors/align_with_label/",
        help="Ding128和标签对齐的内样本"
    )

    parser.add_argument(
        "--Ding128_alb_outer_path",
        type=str,
        default=r"/home/USB_DRIVE3/data_CX/chenx/data_warehouse/Ding_factors/factors_outer/align_with_label/",
        help="Ding128和标签对齐的外样本"
    )

    parser.add_argument(
        "--Ding128_amc_inner_path",
        type=str,
        default=r"/home/USB_DRIVE3/data_CX/chenx/data_warehouse/Ding_factors/align_with_mktcap/",
        help="Ding128和市值对齐的内样本"
    )

    parser.add_argument(
        "--Ding128_amc_outer_path",
        type=str,
        default=r"/home/USB_DRIVE3/data_CX/chenx/data_warehouse/Ding_factors/factors_outer/align_with_mktcap/",
        help="Ding128和市值对齐的外样本"
    )

    parser.add_argument(
        "--Ding128_label_path",
        type=str,
        default=r"/home/hongkou/chenx/data_warehouse/labels/",
        help="Ding128标签路径"
    )

    parser.add_argument(
        "--CY312_factor_path",
        type=str,
        default=r"/home/hongkou/chenx/data_warehouse/CY_1430_factors/factors",
        help="CY312 因子文件夹路径"
    )

    parser.add_argument(
        "--CY312_label_path",
        type=str,
        default=r"/home/hongkou/chenx/data_warehouse/CY_1430_factors/labels",
        help="CY312 因子对应的标签文件夹路径"
    )

    parser.add_argument(
        "--CY312_rollinglbl_align_path",
        type=str,
        default=r"/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_label_align_factor",
        help="CY312 因子滚动训练和标签对齐"
    )

    parser.add_argument(
        "--CY312_rollinglabel_path",
        type=str,
        default=r"/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_label",
        help="CY 因子滚动训练标签"
    )

    parser.add_argument(
        "--CY312_rollingmkt_align_path",
        type=str,
        default=r"/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_market_align_factor",
        help="CY312 因子滚动训练和市值对齐"
    )

    parser.add_argument(
        "--DrJin129_hal_factor_path",
        type=str,
        default=r"/home/USB_DRIVE3/data_CX/chenx/data_warehouse/DrJin_factors/factors/",
        help="DrJin129 因子（半日）文件夹路径"
    )

    parser.add_argument(
        "--DrJin129_all_factor_path",
        type=str,
        default=r"/home/USB_DRIVE3/data_CX/chenx/data_warehouse/DrJin_factors/allday_factors/",
        help="DrJin129 因子（全日）文件夹路径"
    )

    parser.add_argument(
        "--DrJin129_label_path",
        type=str,
        default=r"/home/USB_DRIVE3/data_CX/chenx/data_warehouse/DrJin_factors/labels/",
        help="DrJin129 因子对应的标签文件夹路径"
    )
    parser.add_argument(
        "--DrJin129_rollinglbl_align_path",
        type=str,
        default=r'/home/USB_DRIVE1/Chenx_datawarehouse/DrJin/factors_rolling/r_label_align_factor/',
        help="DrJin129因子滚动训练和标签对齐"
    )

    parser.add_argument(
        "--DrJin129_rollinglabel_path",
        type=str,
        default=r'/home/USB_DRIVE1/Chenx_datawarehouse/DrJin/factors_rolling/r_label/',
        help="DrJin129因子滚动训练标签"
    )

    parser.add_argument(
        "--DrJin129_rollingmkt_align_path",
        type=str,
        default=r'/home/USB_DRIVE1/Chenx_datawarehouse/DrJin/factors_rolling/r_market_align_factor/',
        help="DrJin129因子滚动训练和市值对齐"
    )

    parser.add_argument(
        "--min5_factor_path",
        type=str,
        default=r"/home/hongkou/chenx/data_warehouse/5min_factors/factors/",
        help="5分钟因子文件夹路径"
    )

    parser.add_argument(
        "--min5_label_path",
        type=str,
        default=r"/home/hongkou/chenx/data_warehouse/5min_factors/lables/",
        help="5分钟因子对应的标签文件夹路径"
    )

    return parser.parse_args()

if __name__=='__main__':
    x = get_config()

    print(x.cross_train)
