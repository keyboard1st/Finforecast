import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from DrJin129_load_parquet import *
from torch.utils.data import Subset
import torch
import random

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)


class DrJin129_TimeSeriesDataset(Dataset):
    def __init__(self, sample_set = 'inner',window_size=5, train_data_double_data=None, train_data_label=None):
        # [B,T,C]
        if sample_set in ['inner', 'outer']:
            self.double_data = load_DrJin129_x_parquet(sample_set)
            self.label = load_DrJin129_y_parquet(sample_set)
        # 若为测试集且提供训练集引用
        elif sample_set == 'augmented_test' and train_data_double_data is not None and train_data_label is not None:
            self.double_data = train_data_double_data
            self.label = train_data_label
        else:
            raise ValueError()
        self.num_assets = self.double_data.shape[0]
        self.total_timesteps = self.double_data.shape[1] # 总时间步T
        self.alpha_num = self.double_data.shape[-1] - 1
        self.window_size = window_size
        self.valid_timesteps = self.total_timesteps - self.window_size + 1
        print('total data shape:', self.double_data[:,:,:,0].shape)


    def __len__(self):
        return self.valid_timesteps

    def __getitem__(self, idx):
        X_before = self.double_data[:, idx:idx + self.window_size - 1, :, 1]
        X_now = self.double_data[:, idx + self.window_size - 1, :, 0]
        X_now_expanded = X_now[:, np.newaxis, :]
        X = np.concatenate([X_before, X_now_expanded], axis=1)
        y = self.label[:, idx + self.window_size - 1,0]
        return X, y

def squeeze_collate(batch):
    """当batch_size=1时自动去除批次维度"""
    if len(batch) == 1:
        # 核心修改：转换numpy到Tensor + 压缩批次维度
        x, y = batch[0]
        return (
            torch.tensor(x,dtype=torch.float32).to(device),
            torch.tensor(y,dtype=torch.float32).to(device)
        )
    else:
        raise ValueError('batch_size只能是1')

def get_DrJin129_TimeSeriesDatasetLoader(batchsize = 1, shuffle_time = True, window_size = 5, num_val_windows = 5, val_sample_mode = 'random'):
    '''

    :param batchsize: all_assets代表一个batch是当天全标的数据
    :param shuffle_time: 打乱训练集的时间和标的，但是测试集不变
    :param window_size: 滑动窗口大小
    验证集的设置：取25% - 100%的时间范围内的随机5个窗口
    :return:
    '''

    # 拼接训练集和测试集
    daydata_train = DrJin129_TimeSeriesDataset('inner', window_size)
    train_last_window = daydata_train.double_data[:, -window_size+1:, :, :]
    train_last_window_label = daydata_train.label[:, -window_size+1:, :]
    daydata_test_original = DrJin129_TimeSeriesDataset('outer', window_size)
    augmented_test_data = np.concatenate(
        [train_last_window, daydata_test_original.double_data],
        axis=1
    )
    augmented_test_label = np.concatenate(
        [train_last_window_label, daydata_test_original.label],
        axis=1
    )
    daydata_test = DrJin129_TimeSeriesDataset(sample_set='augmented_test',window_size=window_size,train_data_double_data=augmented_test_data, train_data_label=augmented_test_label)


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
    # test_sampler = SubsetRandomSampler(test_indices) if shuffle_time else None

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

class DrJin129_CrossSectionDataset(Dataset):
    def __init__(self, sample_set = 'inner'):
        self.double_data = load_DrJin129_x_parquet(sample_set)
        self.label = load_DrJin129_y_parquet(sample_set)
        self.data = np.concatenate([self.double_data[:,:,:,0], self.label], axis=-1)
        print('total data shape:', self.data.shape)

    def __getitem__(self, index):
        return self.data[:, index, :]

    def __len__(self):
        return self.data.shape[1]

def get_DrJin129_CrossSectionDatasetLoader(batchsize="all", shuffle_time=False):
    daydata_train = DrJin129_CrossSectionDataset('inner')
    daydata_test = DrJin129_CrossSectionDataset('outer')

    train_indices = list(range(len(daydata_train)))
    test_indices = list(range(len(daydata_test)))

    if isinstance(batchsize, int):
        train_batchsize = test_batchsize = batchsize
    elif batchsize == 'all':
        train_batchsize = len(train_indices)
        test_batchsize = len(test_indices)
    else:
        raise ValueError('你需要将batchsize设置成int整数或者字符串"all"')

    train_sampler = SubsetRandomSampler(train_indices) if shuffle_time else None
    test_sampler = SubsetRandomSampler(test_indices) if shuffle_time else None

    train_dataloader = DataLoader(daydata_train, batch_size=train_batchsize, sampler=train_sampler)
    test_dataloader = DataLoader(daydata_test, batch_size=test_batchsize, sampler=test_sampler)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    # train_dataloader, val_dataloader, test_dataloader = get_JinCSDay_dataloader(batchsize = 1, shuffle_time = True, window_size = 30, num_val_windows = 100, val_sample_mode = 'random')
    #
    # for batch_X, batch_y in train_dataloader:
    #     print(batch_X.shape)
    #     print(batch_y.shape)
    #     valid_mask = ~torch.isnan(batch_y.squeeze())
    #     print(valid_mask.shape)
    #     batch_X = batch_X[valid_mask]
    #     batch_y = batch_y[valid_mask]
    #     batch_X = torch.nan_to_num(batch_X, nan=0)
    #     print(batch_X.shape)
    #     print(batch_y.shape)
    #     break

    train_dataloader, test_dataloader = get_xgb_Jintrainloader(batchsize = 'all', shuffle_time = False)
    for batch in train_dataloader:
        print(batch.shape)
        break

    # train_batch_nums = 0
    # finetune_batch_nums = 0
    # val_batch_nums = 0
    # test_batch_nums = 0
    # for batch_X, batch_y in train_dataloader:
    #     t_batch_X_shape = batch_X.shape
    #     t_batch_y_shape = batch_y.shape
    #     t_type_X = type(batch_X)
    #     train_batch_nums += 1
    # # for batch_X, batch_y in finetune_dataloader:
    # #     f_batch_X_shape = batch_X.shape
    # #     finetune_batch_nums += 1
    # for batch_X, batch_y in val_dataloader:
    #     v_batch_X = batch_X.shape
    #     val_batch_y_shape = batch_y.shape
    #     val_batch_nums += 1
    # for batch_X, batch_y in test_dataloader:
    #     s_batch_X_shape = batch_X.shape
    #     s_batch_y_shape = batch_y.shape
    #     s_type_X = type(batch_X)
    #     test_batch_nums += 1
    #
    # print("train_batch_nums: {:d} | val_batch_nums: {:d} | test_batch_nums: {:d}".format(train_batch_nums, val_batch_nums, test_batch_nums))
    # print("train_batch_X: {} | val_batch_X: {} | test_batch_X: {}".format(t_batch_X_shape, v_batch_X, s_batch_X_shape))
    # print("train_batch_y: {} | val_batch_y: {} | test_batch_y: {}".format(t_batch_y_shape, val_batch_y_shape, s_batch_y_shape))


