import os
import sys
from concurrent import futures

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import random





def load_DrJin129rolling_x_parquet(x_type = 'Jin_factor_lbl_align', sample_set='inner', time_period:str = '2021-2022'):
    from config import get_config
    config = get_config()
    label = pd.read_parquet(os.path.join(config.DrJin129_rollinglabel_path, f'{time_period}/label_{sample_set}.parquet'))

    # if x_type == 'Jin_factor_lbl_align':
    #     halfday_list = sorted([os.path.join(config.DrJin129_rollinglbl_align_path, f'{time_period}/half_F{i}_label_{sample_set}.parquet') for i in range(1, 130)])
    #     allday_list = sorted([os.path.join(config.DrJin129_rollinglbl_align_path, f'{time_period}/all_F{i}_label_{sample_set}.parquet') for i in range(1, 130)])
    # elif x_type == 'Jin_factor_mkt_align':
    halfday_list = sorted([os.path.join(config.DrJin129_rollingmkt_align_path, f'{time_period}/half_F{i}_mkt_{sample_set}.parquet') for i in range(1, 130)])
    allday_list = sorted([os.path.join(config.DrJin129_rollingmkt_align_path, f'{time_period}/all_F{i}_mkt_{sample_set}.parquet') for i in range(1, 130)])
    # else:
    #     raise NotImplementedError

    with futures.ThreadPoolExecutor(max_workers=24) as executor:
        if x_type == 'Jin_factor_lbl_align':
            halfday_dfs = list(executor.map(lambda f: pd.read_parquet(f, columns=label.columns),halfday_list))
        elif x_type == 'Jin_factor_mkt_align':
            halfday_dfs = list(executor.map(lambda f: pd.read_parquet(f),halfday_list))
    with futures.ThreadPoolExecutor(max_workers=24) as executor:
        if x_type == 'Jin_factor_lbl_align':
            allday_dfs = list(executor.map(lambda f: pd.read_parquet(f, columns=label.columns),allday_list))
        elif x_type == 'Jin_factor_mkt_align':
            allday_dfs = list(executor.map(lambda f: pd.read_parquet(f),allday_list))

    def standard(df):
        row_std = df.std(axis=1)
        zero_std_mask = (row_std == 0)
        if zero_std_mask.any():
            zero_std_indices = row_std.index[zero_std_mask]
            print(f"发现 {len(zero_std_indices)} 行标准差为0，行索引为: {zero_std_indices.tolist()}")

            row_std = row_std.where(row_std != 0, 1)
        return (df.sub(df.mean(axis=1), axis=0)).div(row_std, axis=0)

    halfday_arr = np.array([standard(future).values for future in halfday_dfs]).transpose(2, 1, 0)
    allday_arr = np.array([standard(future).values for future in allday_dfs]).transpose(2, 1, 0)

    double_data = np.stack((halfday_arr, allday_arr), axis=-1)
    return double_data

def load_DrJin129rolling_y_parquet(y_type = 'Jin_label', sample_set = 'inner', time_period:str = '2021-2022'):
    from config import get_config
    config = get_config()

    def standard(df):
        return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)
    if y_type == 'Jin_label':
        return np.expand_dims(standard(pd.read_parquet(os.path.join(config.DrJin129_rollinglabel_path, f'{time_period}/label_{sample_set}.parquet'))).values.transpose(1, 0), axis=2)


class DrJin129_rollingtrain_TimeSeriesDataset(Dataset):
    def __init__(self, sample_set = 'inner', window_size=5, time_period:str = '2021-2022', train_data_double_data=None, train_data_label=None):
        # [B,T,C]
        if sample_set in ['inner', 'outer']:
            self.double_data = load_DrJin129rolling_x_parquet('Jin_factor_lbl_align',sample_set, time_period)
            self.label = load_DrJin129rolling_y_parquet('Jin_label',sample_set, time_period)
            self.double_data = self.double_data[:,:self.label.shape[1], :, :]


        elif sample_set == 'augmented_test' and train_data_double_data is not None and train_data_label is not None:

            self.double_data = train_data_double_data
            self.label = train_data_label
        else:
            raise NotImplementedError
        self.num_assets = self.double_data.shape[0]
        self.total_timesteps = self.double_data.shape[1]  # 总时间步T
        self.alpha_num = self.double_data.shape[-1] - 1
        self.window_size = window_size
        self.valid_timesteps = self.total_timesteps - self.window_size + 1
        print('double_data shape:', self.double_data[:, :, :, 0].shape)
        print('label shape:', self.label[:, :, 0].shape)

    def __len__(self):
        return self.valid_timesteps

    def __getitem__(self, idx):
        X_before = self.double_data[:, idx:idx + self.window_size - 1, :, 1]
        X_now = self.double_data[:, idx + self.window_size - 1, :, 0]
        X_now_expanded = X_now[:, np.newaxis, :]
        X = np.concatenate([X_before, X_now_expanded], axis=1)
        y = self.label[:, idx + self.window_size - 1, 0]
        return X, y


def get_DrJin129_rollingtrain_TimeSeriesLoader(batchsize = 1, shuffle_time = True, window_size = 30, num_val_windows = 100, val_sample_mode = 'random', time_period:str = '2021-2022', config = None):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    random_seed = config.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    print('random_seed:', random_seed)

    # 拼接训练集和测试集
    daydata_train = DrJin129_rollingtrain_TimeSeriesDataset('inner', window_size, time_period=time_period)
    train_last_window = daydata_train.double_data[:, -window_size + 1:, :, :]
    train_last_window_label = daydata_train.label[:, -window_size + 1:, :]
    daydata_test_original = DrJin129_rollingtrain_TimeSeriesDataset('outer', window_size, time_period=time_period)
    augmented_test_data = np.concatenate(
        [train_last_window, daydata_test_original.double_data],
        axis=1
    )
    augmented_test_label = np.concatenate(
        [train_last_window_label, daydata_test_original.label],
        axis=1
    )
    daydata_test = DrJin129_rollingtrain_TimeSeriesDataset(sample_set='augmented_test', window_size=window_size,train_data_double_data=augmented_test_data, train_data_label=augmented_test_label)

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


class DrJin129_rollingfintest_TimeSeriesDataset(Dataset):
    def __init__(self, sample_set = 'inner', window_size=5, time_period:str = '2021-2022', train_data_ref=None):
        # [B,T,C]
        if sample_set == 'inner' or sample_set == 'outer':
            self.double_data = load_DrJin129rolling_x_parquet('Jin_factor_mkt_align', sample_set, time_period)

        elif sample_set == 'augmented_test' and train_data_ref is not None:
            self.double_data = train_data_ref
        else:
            raise NotImplementedError
        self.num_assets = self.double_data.shape[0]
        self.total_timesteps = self.double_data.shape[1]  # 总时间步T
        self.alpha_num = self.double_data.shape[-1] - 1
        self.window_size = window_size
        self.valid_timesteps = self.total_timesteps - self.window_size + 1
        print('total data shape:', self.double_data[:, :, :, 0].shape)

    def __len__(self):
        return self.valid_timesteps

    def __getitem__(self, idx):
        X_before = self.double_data[:, idx:idx + self.window_size - 1, :, 1]
        X_now = self.double_data[:, idx + self.window_size - 1, :, 0]
        X_now_expanded = X_now[:, np.newaxis, :]
        X = np.concatenate([X_before, X_now_expanded], axis=1)
        return X

def get_DrJin129_rollingfintest_TimeSeriesLoader(batchsize = 1, window_size = 30, time_period:str = '2021-2022', config = None):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 拼接训练集和测试集
    daydata_train = DrJin129_rollingfintest_TimeSeriesDataset('inner', window_size, time_period=time_period)
    train_last_window = daydata_train.double_data[:, -window_size+1:, :]
    daydata_test_original = DrJin129_rollingfintest_TimeSeriesDataset('outer', window_size, time_period=time_period)
    augmented_test_data = np.concatenate(
        [train_last_window, daydata_test_original.double_data],
        axis=1
    )
    daydata_test = DrJin129_rollingfintest_TimeSeriesDataset(sample_set='augmented_test',window_size=window_size,train_data_ref=augmented_test_data)

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
    test_dataloader = DataLoader(daydata_test, batch_size=batchsize,collate_fn=squeeze_collate)

    return test_dataloader


class DrJin129_rollingtrain_CrossSectionDataset(Dataset):
    def __init__(self, sample_set = 'inner', time_period:str = '2021-2022'):
        self.double_data = load_DrJin129rolling_x_parquet('Jin_factor_lbl_align', sample_set, time_period)
        self.label = load_DrJin129rolling_y_parquet('Jin_label', sample_set, time_period)
        # 因为outer的label长于double_data，所以需要截断outer的double_data的长度
        self.data = np.concatenate([self.double_data[:,:self.label.shape[1], :, 0], self.label], axis=-1)
        print('total data shape:', self.data.shape)

    def __getitem__(self, index):
        return self.data[:, index, :]

    def __len__(self):
        return self.data.shape[1]

def get_DrJin129_rollingtrain_CrossSectionDatasetLoader(batchsize="all", shuffle_time=False, time_period:str = '2021-2022'):
    daydata_train = DrJin129_rollingtrain_CrossSectionDataset('inner', time_period)
    daydata_test = DrJin129_rollingtrain_CrossSectionDataset('outer',time_period)

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

class DrJin129_rollingfintest_CrossSectionDataset(Dataset):
    def __init__(self, sample_set = 'inner', time_period:str = '2021-2022'):
        self.double_data = load_DrJin129rolling_x_parquet('Jin_factor_mkt_align',sample_set=sample_set,time_period=time_period)
        self.data = self.double_data[:, :, :, 0]
        print('total data shape:', self.data.shape)

    def __getitem__(self, index):
        return self.data[:, index, :]

    def __len__(self):
        return self.data.shape[1]

def get_DrJin129_rollingfintest_CrossSectionDatasetLoader(time_period:str = '2021-2022'):
    daydata_test = DrJin129_rollingfintest_CrossSectionDataset('outer',time_period)
    test_dataloader = DataLoader(daydata_test, batch_size=len(daydata_test))
    return test_dataloader

if __name__ == '__main__':
    # train_loader, test_loader = get_xgb_rollingtrainloader()
    # testset = next(iter(test_loader))
    # print(testset.shape)
    # print(testset[:5,:5,0])
    # fin_testloader = get_xgb_fintestloader()
    # fin_testset = next(iter(fin_testloader))
    # print(fin_testset.shape)
    # print(fin_testset[:5,:5,0])

    train_loader, val_loader, test_loader = get_CSDay_rollingtrainloader()
    print(next(iter(train_loader)).shape)
    print(next(iter(val_loader)).shape)
    print(next(iter(test_loader)).shape)

    # train_set = next(iter(xgb_trainloader))
    # test_set = next(iter(xgb_testloader))
    # print(train_set.shape, test_set.shape)
    # train_data = train_set.reshape(-1, train_set.shape[-1])
    # mask = ~torch.isnan(train_data[:, -1])
    # train_data = train_data[mask]
    # train_nan_mask = torch.isnan(train_data)
    # train_nan_counts = train_nan_mask.sum(dim=1)
    # train_indices = torch.where(train_nan_counts <= 8)[0]
    #
    # train_data = train_data[train_indices]
    # train_data = np.array(train_data)
    # train_data_x = train_data[:, :-1]
    # train_data_y = train_data[:, -1]
    # print(train_data_x.shape, train_data_y.shape)

