import os
from concurrent import futures

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from CY312_load_parquet import load_x_CY312_parquet, load_y_CY312_parquet


class CY312_CrossSectionDataset(Dataset):
    def __init__(self, sample_set='inner'):
        self.data = np.concatenate(
            [load_x_CY312_parquet(sample_set), load_y_CY312_parquet(sample_set)], axis=-1)
        print('total data shape:', self.data.shape)

    def __getitem__(self, index):
        return self.data[:, index, :]

    def __len__(self):
        return self.data.shape[1]


def get_CY312_CrossSectionDataloader(batchsize='all',shuffle_time=False):
    daydata_train = CY312_CrossSectionDataset('inner')
    daydata_test = CY312_CrossSectionDataset('outer')

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
