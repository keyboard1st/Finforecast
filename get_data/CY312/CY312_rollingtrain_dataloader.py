import numpy as np
import pandas as pd
import os
from concurrent import futures

from sympy.simplify.simplify import factor_sum
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.data import Subset
import torch
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def standard(df):
    return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

# def load_CY312rolling_x_parquet(x_type='CY_factor_lbl_align', sample_set = 'inner', time_period:str = '2021-2022'):
#     from config import get_config
#     config = get_config()
#     label = pd.read_parquet(os.path.join(config.CY312_rollinglabel_path, f'{time_period}/label_{sample_set}.parquet'))
#     file_list = [os.path.join(config.CY312_rollingmkt_align_path, f'{time_period}/F{i}_mkt_{sample_set}.parquet') for i in range(1, 313)]
#     # if x_type == 'CY_factor_lbl_align':
#     #     file_list = [os.path.join(config.CY312_rollinglbl_align_path, f'{time_period}/F{i}_label_{sample_set}.parquet') for i in range(1, 313)]
#     # elif x_type == 'CY_factor_mkt_align':
#     #     file_list = [os.path.join(config.CY312_rollingmkt_align_path, f'{time_period}/F{i}_mkt_{sample_set}.parquet') for i in range(1, 313)]
#     # else:
#     #     raise NotImplementedError
#
#     with futures.ThreadPoolExecutor(max_workers=24) as executor:
#         if x_type == 'CY_factor_lbl_align':
#             factors = list(executor.map(lambda f: pd.read_parquet(f, columns=label.columns), file_list))
#         elif x_type == 'CY_factor_mkt_align':
#             factors = list(executor.map(lambda f: pd.read_parquet(f), file_list))
#     factor_arr = np.array([standard(future).values for future in factors]).transpose(2, 1, 0)
#     return factor_arr

def load_CY312rolling_x_parquet(x_type='CY_factor_lbl_align', sample_set = 'inner', time_period:str = '2021-2022'):
    from config import get_config
    config = get_config()
    label = pd.read_parquet(os.path.join(config.CY312_rollinglabel_path, f'{time_period}/label_{sample_set}.parquet'))
    target_cols = label.columns
    file_list = [os.path.join(config.CY312_rollingmkt_align_path, f'{time_period}/F{i}_mkt_{sample_set}.parquet') for i in range(1, 313)]

    with futures.ThreadPoolExecutor(max_workers=24) as executor:
        if x_type == 'CY_factor_lbl_align':
            def _load_and_proc(path):
                df = pd.read_parquet(path)  # 读入全部列
                df = standard(df)  # 先行标准化
                return df.loc[:, target_cols]

            processed = list(executor.map(_load_and_proc, file_list))
        elif x_type == 'CY_factor_mkt_align':
            processed = list(executor.map(pd.read_parquet, file_list))
    factor_arr = np.array([df.values for df in processed]).transpose(2, 1, 0)
    return factor_arr

def load_CY312rolling_y_parquet(y_type = 'CY_label', sample_set = 'inner', time_period:str = '2021-2022'):
    from config import get_config
    config = get_config()

    if y_type == 'CY_label':
        return np.expand_dims(standard(pd.read_parquet(os.path.join(config.CY312_rollinglabel_path, f'{time_period}/label_{sample_set}.parquet'))).values.transpose(1, 0), axis=2)


class CY312_rollingtrain_TimeSeriesDataset(Dataset):
    def __init__(self, sample_set = 'inner', window_size=5, time_period:str = '2021-2022', train_data_ref=None):
        # [B,T,C]
        if sample_set == 'inner' or sample_set == 'outer':
            self.data = np.concatenate([load_CY312rolling_x_parquet('CY_factor_lbl_align',sample_set=sample_set,time_period=time_period), load_CY312rolling_y_parquet(y_type = 'CY_label',sample_set=sample_set,time_period=time_period)], axis=-1)

        elif sample_set == 'augmented_test' and train_data_ref is not None:
            self.data = train_data_ref
        else:
            raise NotImplementedError
        self.total_timesteps = self.data.shape[1] # 总时间步T
        self.window_size = window_size
        self.valid_timesteps = self.total_timesteps - self.window_size + 1
        print('total data shape:', self.data.shape)

    def __len__(self):
        return self.valid_timesteps

    def __getitem__(self, idx):
        X = self.data[:, idx:idx + self.window_size, :-1]
        y = self.data[:, idx + self.window_size - 1, -1]
        return X, y

def get_CY312_rollingtrain_TimeSeriesLoader(batchsize = 1, shuffle_time = True, window_size = 30, num_val_windows = 100, val_sample_mode = 'random', time_period:str = '2021-2022', config = None):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 拼接训练集和测试集
    daydata_train = CY312_rollingtrain_TimeSeriesDataset('inner', window_size, time_period=time_period)
    train_last_window = daydata_train.data[:, -window_size+1:, :]
    daydata_test_original = CY312_rollingtrain_TimeSeriesDataset('outer', window_size, time_period=time_period)
    augmented_test_data = np.concatenate(
        [train_last_window, daydata_test_original.data],
        axis=1
    )
    daydata_test = CY312_rollingtrain_TimeSeriesDataset(sample_set='augmented_test',window_size=window_size,train_data_ref=augmented_test_data)

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

class CY312_rollingfintest_TimeSeriesDataset(Dataset):
    def __init__(self, sample_set = 'inner', window_size=5, time_period:str = '2021-2022', train_data_ref=None):
        # [B,T,C]
        if sample_set == 'inner' or sample_set == 'outer':
            self.data = load_CY312rolling_x_parquet('CY_factor_mkt_align',sample_set=sample_set,time_period=time_period)

        elif sample_set == 'augmented_test' and train_data_ref is not None:
            self.data = train_data_ref
        else:
            raise NotImplementedError
        self.total_timesteps = self.data.shape[1] # 总时间步T
        self.window_size = window_size
        self.valid_timesteps = self.total_timesteps - self.window_size + 1
        print('total data shape:', self.data.shape)

    def __len__(self):
        return self.valid_timesteps

    def __getitem__(self, idx):
        X = self.data[:, idx:idx + self.window_size,:]
        return X

def get_CY312_rollingfintest_TimeSeriesloader(batchsize = 1, window_size = 30, time_period:str = '2021-2022', config=None):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 拼接训练集和测试集
    daydata_train = CY312_rollingfintest_TimeSeriesDataset('inner', window_size, time_period=time_period)
    train_last_window = daydata_train.data[:, -window_size+1:, :]
    daydata_test_original = CY312_rollingfintest_TimeSeriesDataset('outer', window_size, time_period=time_period)
    augmented_test_data = np.concatenate(
        [train_last_window, daydata_test_original.data],
        axis=1
    )
    daydata_test = CY312_rollingfintest_TimeSeriesDataset(sample_set='augmented_test',window_size=window_size,train_data_ref=augmented_test_data)

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

class CY312_rollingtrain_CrossSectionDataset(Dataset):
    def __init__(self, sample_set = 'inner', time_period:str = '2021-2022'):
        self.data = np.concatenate([load_CY312rolling_x_parquet('CY_factor_lbl_align',sample_set=sample_set,time_period=time_period), load_CY312rolling_y_parquet(y_type = 'CY_label',sample_set=sample_set,time_period=time_period)], axis=-1)
        print('total data shape:', self.data.shape)

    def __getitem__(self, index):
        return self.data[:, index, :]

    def __len__(self):
        return self.data.shape[1]

def get_CY312_rollingtrain_CrossSectionLoader(batchsize="all", shuffle_time=False, time_period:str = '2021-2022'):
    daydata_train = CY312_rollingtrain_CrossSectionDataset('inner', time_period)
    daydata_test = CY312_rollingtrain_CrossSectionDataset('outer',time_period)

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

class CY312_rollingfintest_CrossSectionDataset(Dataset):
    def __init__(self, sample_set = 'inner', time_period:str = '2021-2022'):
        self.data = load_CY312rolling_x_parquet('CY_factor_mkt_align',sample_set=sample_set,time_period=time_period)
        print('total data shape:', self.data.shape)

    def __getitem__(self, index):
        return self.data[:, index, :]

    def __len__(self):
        return self.data.shape[1]

def get_CY312_rollingfintest_CrossSectionLoader(time_period:str = '2021-2022'):
    daydata_test = CY312_rollingfintest_CrossSectionDataset('outer',time_period)
    test_dataloader = DataLoader(daydata_test, batch_size=len(daydata_test))
    return test_dataloader


if __name__ == '__main__':
    train_loader, test_loader = get_xgb_rollingtrainloader()
    testset = next(iter(test_loader))
    print(testset.shape)
    print(testset[:5,:5,0])
    fin_testloader = get_xgb_fintestloader()
    fin_testset = next(iter(fin_testloader))
    print(fin_testset.shape)
    print(fin_testset[:5,:5,0])

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

