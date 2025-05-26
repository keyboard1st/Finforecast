import os
import sys
sys.path.append(os.path.dirname(__file__))
from load_minute_parquet import load_10minute_rolling_x_parquet, load_10minute_rolling_y_parquet
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.data import Subset
import torch
import random

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


class Minute_TimeSeries_Dataset(Dataset):
    def __init__(self, sample_set = 'inner', min_freq = 10, window_size=4, time_period = '2021-2022', train_features_ref=None, train_label_ref=None):
        # [B,T,C]
        if sample_set in ['inner', 'outer']:
            self.features = load_10minute_rolling_x_parquet(x_type = 'lbl_align',sample_set=sample_set, time_period=time_period)  # X必须shift掉最后半小时数据
            self.label = load_10minute_rolling_y_parquet(sample_set=sample_set, time_period=time_period)
        # 若为测试集且提供训练集引用
        elif sample_set == 'augmented_test' and train_features_ref is not None:
            self.features = train_features_ref
            self.label = train_label_ref
        self.bars_num_oneday  = 240 // min_freq
        # assert self.features.shape[1] // self.bars_num_oneday == self.label.shape[1]
        self.total_timesteps = self.features.shape[1] // self.bars_num_oneday
        self.window_size = window_size
        self.valid_timesteps = self.total_timesteps - self.window_size + 1
        print('total features shape:', self.features.shape)
        print('label shape:', self.label.shape)

    def __len__(self):
        return self.valid_timesteps

    def __getitem__(self, day_idx):
        start_idx = day_idx * self.bars_num_oneday
        end_idx = start_idx + self.window_size * self.bars_num_oneday
        X = self.features[:, start_idx:end_idx,:]
        y = self.label[:, day_idx + self.window_size - 1, 0]
        return X, y

def get_min10_rollingtrain_TimeSeriesLoader(batchsize = 1, shuffle_time = True, window_size = 4, num_val_windows = 100, val_sample_mode = 'random', time_period:str = '2021-2022', config = None):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 拼接训练集和测试集
    daydata_train = Minute_TimeSeries_Dataset('inner', window_size = window_size, time_period=time_period)
    train_features_last_window = daydata_train.features[:, -(window_size-1)*24:, :]
    train_label_last_window = daydata_train.label[:, -window_size+1:, :]
    daydata_test_original = Minute_TimeSeries_Dataset('outer', window_size = window_size, time_period=time_period)
    augmented_test_features = np.concatenate([train_features_last_window, daydata_test_original.features],axis=1)
    augmented_test_label = np.concatenate([train_label_last_window, daydata_test_original.label],axis=1)
    daydata_test = Minute_TimeSeries_Dataset(sample_set='augmented_test',window_size=window_size, train_features_ref=augmented_test_features, train_label_ref=augmented_test_label)

    valid_timesteps = daydata_train.valid_timesteps  # 总时间窗口数
    if val_sample_mode == 'random':
        # 随机选验证集窗口
        start_in_tail = valid_timesteps // 4  # 25%位置
        end_in_tail = valid_timesteps - 1  # 100%位置（最后一个时间窗口）
        candidate_windows = list(range(start_in_tail, end_in_tail + 1))
        valid_indices = random.sample(candidate_windows, num_val_windows)
    elif val_sample_mode == 'tail':
        # 取训练集尾部10%的验证集窗口
        start_in_tail = valid_timesteps - valid_timesteps // 10
        end_in_tail = valid_timesteps - 1  # 100%位置（最后一个时间窗口）
        candidate_windows = list(range(start_in_tail, end_in_tail + 1))
        valid_indices = candidate_windows
    else:
        raise ValueError('val_sample_mode只能是random或者tail')

    print(f"Vali Selected windows: {valid_indices}")

    test_indices  = list(range(len(daydata_test)))
    train_indices = list(set(range(len(daydata_train))) - set(valid_indices))
    # 创建子集
    train_dataset = Subset(daydata_train, train_indices)
    val_dataset = Subset(daydata_train, valid_indices)
    test_dataset = daydata_test
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")
    print(f"Test Dataset size: {len(test_dataset)}")

    train_subset_indices = list(range(len(train_dataset)))

    if isinstance(batchsize, int):
        train_batchsize = test_batchsize = batchsize
    elif batchsize == 'all':
        # 一次获得所有数据
        train_batchsize = len(train_indices)
        test_batchsize = len(test_indices)
    else:
        raise ValueError('你需要将batchsize设置成int整数或者字符串"all"')

    if shuffle_time:
        random.shuffle(train_subset_indices)
    train_sampler = SubsetRandomSampler(train_subset_indices) if shuffle_time else None
    def squeeze_collate(batch):
        """当batch_size=1时自动去除批次维度"""
        if len(batch) == 1:
            # 核心修改：转换numpy到Tensor + 压缩批次维度
            x, y = batch[0]
            return (
                torch.tensor(x, dtype=torch.float32).to(device),
                torch.tensor(y, dtype=torch.float32).to(device)
            )
        else:
            raise ValueError('batch_size只能是1')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, sampler=train_sampler,collate_fn=squeeze_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=train_batchsize,collate_fn=squeeze_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batchsize,collate_fn=squeeze_collate)
    for batch_X, _ in train_dataloader:
        train_shape = batch_X.shape  # 形状为 [batch_size, window_size, num_features]
        break
    for batch_X, _ in val_dataloader:
        val_shape = batch_X.shape
        break
    for batch_X, _ in test_dataloader:
        test_shape = batch_X.shape
        break

    print("train_dataloader: {:d} | Validation dataloader: {:d} | Test dataloader: {:d}".format(len(train_dataloader), len(val_dataloader), len(test_dataloader)))
    print("train_batchsize: {:d} | Validation batch size: {:d} | Test batch size: {:d}".format(train_batchsize, train_batchsize, test_batchsize))
    print("train_batch_X: {} | val_batch_X: {} | test_batch_X: {}".format(train_shape, val_shape, test_shape))


    return train_dataloader, val_dataloader, test_dataloader

class min10_rollingfintest_TimeSeriesDataset(Dataset):
    def __init__(self, sample_set = 'inner', window_size=5, min_freq=10, time_period:str = '2021-2022', train_data_ref=None):
        # [B,T,C]
        if sample_set == 'inner' or sample_set == 'outer':
            self.data = load_10minute_rolling_x_parquet('mkt_align',sample_set=sample_set,time_period=time_period)

        elif sample_set == 'augmented_test' and train_data_ref is not None:
            self.data = train_data_ref
        else:
            raise NotImplementedError
        self.bars_num_oneday = 240 // min_freq
        self.total_timesteps = self.data.shape[1] // self.bars_num_oneday
        self.window_size = window_size
        self.valid_timesteps = self.total_timesteps - self.window_size + 1
        print('total data shape:', self.data.shape)

    def __len__(self):
        return self.valid_timesteps

    def __getitem__(self, day_idx):
        start_idx = day_idx * self.bars_num_oneday
        end_idx = start_idx + self.window_size * self.bars_num_oneday
        X = self.data[:, start_idx:end_idx,:]
        return X

def get_min10_rollingfintest_TimeSeriesloader(batchsize = 1, window_size = 4, time_period:str = '2021-2022', config=None):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 拼接训练集和测试集
    daydata_train = min10_rollingfintest_TimeSeriesDataset('inner', window_size, time_period=time_period)
    train_last_window = daydata_train.data[:, -(window_size-1)*24:, :]
    daydata_test_original = min10_rollingfintest_TimeSeriesDataset('outer', window_size, time_period=time_period)
    augmented_test_data = np.concatenate(
        [train_last_window, daydata_test_original.data],
        axis=1
    )
    daydata_test = min10_rollingfintest_TimeSeriesDataset(sample_set='augmented_test',window_size=window_size,train_data_ref=augmented_test_data)

    def squeeze_collate(batch):
        # 检查 batch 中的样本类型
        sample = batch[0]
        if isinstance(sample, tuple):
            x_list, y_list = zip(*batch)
            x_tensor = torch.tensor(np.stack(x_list), dtype=torch.float32).to(device)
            y_tensor = torch.tensor(np.stack(y_list), dtype=torch.float32).to(device)
            x_tensor = x_tensor.squeeze(0)
            y_tensor = y_tensor.squeeze(0)
            return x_tensor, y_tensor
        else:
            x_tensor = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)
            x_tensor = x_tensor.squeeze(0)

            return x_tensor
    test_dataloader = DataLoader(daydata_test, batch_size=batchsize, collate_fn=squeeze_collate)

    return test_dataloader


if __name__ == '__main__':
    from config import get_config
    config = get_config()
    test_dataloader = get_min10_rollingfintest_TimeSeriesloader(batchsize = 1, window_size = 4, time_period = '2021-2022', config=config)
    for i, x in enumerate(test_dataloader):
        print(x.shape)
        break
