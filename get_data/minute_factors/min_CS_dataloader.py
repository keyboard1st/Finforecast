import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from load_minute_parquet import load_x_parquet, load_y_parquet
from torch.utils.data import Subset
import torch
import random

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

class Minute_TimeSeries_Dataset(Dataset):
    def __init__(self, sample_set = 'inner', min_freq = 10, window_size=4, train_data_ref=None):
        # [B,T,C]
        if sample_set in ['inner', 'outer']:
            self.features = load_x_parquet(sample_set)  # X必须shift掉最后半小时数据
            self.label = load_y_parquet(sample_set)
        # 若为测试集且提供训练集引用
        elif sample_set == 'augmented_test' and train_data_ref is not None:
            self.data = train_data_ref
        self.bars_num_oneday  = 240 // min_freq
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
        y = self.data[:, day_idx + self.window_size - 1, 0]
        return X, y

if __name__ == '__main__':
    train_dataset = CSDayDataset(sample_set='inner', window_size=5)