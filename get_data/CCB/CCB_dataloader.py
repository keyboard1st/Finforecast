"""
CCB DataLoader - 最终统一版本

Author: Keyboardist
Date: 2025-06-30
"""

import os
from concurrent import futures
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
from get_data.CCB.CCB_load_parquet import load_x_CCB_parquet, load_y_CCB_parquet

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


# ============================================================================
# 简化配置类
# ============================================================================

class CCBDataLoaderConfig:
    """简化的CCB DataLoader配置类"""
    
    def __init__(self, 
                 data_type='timeseries',        # 'timeseries' or 'crosssection' 
                 include_labels=True,           # True: 包含y标签, False: 只有X (market模式)
                 
                 # 通用参数
                 batchsize=1,
                 train_time_period='2019-2025', 
                 test_time_period='2025-2026',
                 shuffle_time=True, 
                 
                 # TimeSeries 专用参数
                 window_size=30, 
                 val_sample_mode='random',
                 num_val_windows=100,
                 num_val_days=100,
                 
                 # 设备配置
                 device_config=None):
        
        self.data_type = data_type
        self.include_labels = include_labels
        
        # 通用参数
        self.batchsize = batchsize
        self.train_time_period = train_time_period
        self.test_time_period = test_time_period
        self.shuffle_time = shuffle_time
        
        # TimeSeries 专用参数
        self.window_size = window_size
        self.val_sample_mode = val_sample_mode
        self.num_val_windows = num_val_windows
        self.num_val_days = num_val_days
        
        # 设备配置
        self.device = torch.device(device_config.device if device_config and torch.cuda.is_available() else "cpu")
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数"""
        if self.data_type not in ['timeseries', 'crosssection']:
            raise ValueError('data_type must be timeseries or crosssection')
        
        if not isinstance(self.include_labels, bool):
            raise ValueError('include_labels must be True or False')
        
        if self.data_type == 'timeseries':
            if self.val_sample_mode not in ['random', 'tail', 'random_days']:
                raise ValueError('val_sample_mode can only be random, tail or random_days')
            
            if self.val_sample_mode == 'random_days' and self.num_val_days <= 0:
                raise ValueError('num_val_days must be positive for random_days mode')
            
            if self.window_size <= 0:
                raise ValueError('window_size must be positive')
    
    def __str__(self):
        """友好的字符串表示"""
        mode = "Market" if not self.include_labels else "Standard"
        return f"{mode} {self.data_type.title()}"


# ============================================================================
# 数据集类定义
# ============================================================================

class CCB_TimeSeriesDataset(Dataset):
    """时序数据集（包含X和y）"""
    def __init__(self, time_period='2019-2025', window_size=5, train_data_ref=None):
        # [B,T,C]
        if time_period not in ['augmented_test', 'augmented_train', 'augmented_val']:
            self.data = np.concatenate([load_x_CCB_parquet(time_period), load_y_CCB_parquet(time_period)], axis=-1)
        # If augmented data and data reference is provided
        elif time_period in ['augmented_test', 'augmented_train', 'augmented_val'] and train_data_ref is not None:
            self.data = train_data_ref
        self.num_assets = self.data.shape[0]
        self.total_timesteps = self.data.shape[1] # Total timesteps T
        self.alpha_num = self.data.shape[-1] - 1
        self.window_size = window_size
        self.valid_timesteps = self.total_timesteps - self.window_size + 1
        print('total data shape:', self.data.shape)

    def __len__(self):
        return self.valid_timesteps

    def __getitem__(self, idx):
        X = self.data[:, idx:idx + self.window_size, :-1]
        y = self.data[:, idx + self.window_size - 1, -1]
        return X, y


class CCB_MarketDataset(Dataset):
    """市场数据集（只有X，用于预测）"""
    def __init__(self, time_period='2019-2025', window_size=5, train_data_ref=None):
        # [B,T,C] - Only load feature data, no labels
        if time_period not in ['augmented_test', 'augmented_train', 'augmented_val']:
            self.data = load_x_CCB_parquet(time_period)  # Only X data
        # If augmented data and data reference is provided
        elif time_period in ['augmented_test', 'augmented_train', 'augmented_val'] and train_data_ref is not None:
            self.data = train_data_ref
        self.num_assets = self.data.shape[0]
        self.total_timesteps = self.data.shape[1] # Total timesteps T
        self.alpha_num = self.data.shape[-1]  # No -1 because no labels
        self.window_size = window_size
        self.valid_timesteps = self.total_timesteps - self.window_size + 1
        print('total data shape:', self.data.shape)

    def __len__(self):
        return self.valid_timesteps

    def __getitem__(self, idx):
        # Only return X data, no y
        X = self.data[:, idx:idx + self.window_size, :]
        return X


class CCB_CrossSectionDataset(Dataset):
    """截面数据集（包含X和y）"""
    def __init__(self, time_period='2019-2025'):
        self.data = np.concatenate(
            [load_x_CCB_parquet(time_period), load_y_CCB_parquet(time_period)], axis=-1)
        print('total data shape:', self.data.shape)

    def __getitem__(self, index):
        return self.data[:, index, :]

    def __len__(self):
        return self.data.shape[1]


class CCB_MKT_CrossSectionDataset(Dataset):
    """市场截面数据集（只有X）"""
    def __init__(self, time_period='2019-2025'):
        self.data = load_x_CCB_parquet(time_period)
        print('total data shape:', self.data.shape)

    def __getitem__(self, index):
        return self.data[:, index, :]

    def __len__(self):
        return self.data.shape[1]


# ============================================================================
# 验证集生成策略
# ============================================================================

class ValidationSetGenerator:
    """处理不同的验证集生成策略"""
    
    @staticmethod
    def generate_random_windows(train_dataset, config):
        """生成随机窗口验证集"""
        valid_timesteps = train_dataset.valid_timesteps
        start_in_tail = valid_timesteps // 4  # 25% position
        end_in_tail = valid_timesteps - 1  # 100% position (last time window)
        candidate_windows = list(range(start_in_tail, end_in_tail + 1))
        valid_indices = random.sample(candidate_windows, config.num_val_windows)
        return valid_indices, None, None
    
    @staticmethod
    def generate_tail_windows(train_dataset, config):
        """生成尾部窗口验证集"""
        valid_timesteps = train_dataset.valid_timesteps
        start_in_tail = valid_timesteps - valid_timesteps // 10
        end_in_tail = valid_timesteps - 1  # 100% position (last time window)
        candidate_windows = list(range(start_in_tail, end_in_tail + 1))
        valid_indices = candidate_windows
        return valid_indices, None, None
    
    @staticmethod
    def generate_random_days(train_dataset, config):
        """生成随机天数拼接验证集（避免数据泄露）"""
        print(f"\n=== Using random_days mode to generate validation set ===")
        full_data = train_dataset.data  # [B, T, C]
        total_days = full_data.shape[1]
        
        print(f"Total days: {total_days}")
        print(f"Validation days: {config.num_val_days}")
        
        if config.num_val_days >= total_days:
            raise ValueError(f"Validation days ({config.num_val_days}) cannot be greater than or equal to total days ({total_days})")
        
        # Randomly select validation days (completely random, non-consecutive)
        all_day_indices = list(range(total_days))
        val_day_indices = sorted(random.sample(all_day_indices, config.num_val_days))
        train_day_indices = sorted([day for day in all_day_indices if day not in val_day_indices])
        
        print(f"Randomly selected validation days: {val_day_indices[:10]}{'...' if len(val_day_indices) > 10 else ''}")
        print(f"Validation days range: [{min(val_day_indices)}, {max(val_day_indices)}]")
        print(f"Training days count: {len(train_day_indices)}")
        
        # Concatenate validation data: concatenate in selected day order
        val_data = full_data[:, val_day_indices, :]  # [B, num_val_days, C]
        
        # Concatenate training data: concatenate in chronological order
        train_data = full_data[:, train_day_indices, :]  # [B, remaining_days, C]
        
        print(f"Concatenated training data shape: {train_data.shape}")
        print(f"Concatenated validation data shape: {val_data.shape}")
        
        # Check if validation set has enough days to generate windows
        val_windows = val_data.shape[1] - config.window_size + 1
        if val_windows <= 0:
            raise ValueError(f"Validation days ({config.num_val_days}) insufficient to generate windows, need at least {config.window_size} days")
        
        print(f"Validation windows can be generated: {val_windows}")
        print("✅ Training and validation days completely isolated, no overlap")
        
        # Recreate datasets based on concatenated data
        train_dataset_new = CCB_TimeSeriesDataset(time_period='augmented_train', window_size=config.window_size, train_data_ref=train_data)
        val_dataset_new = CCB_TimeSeriesDataset(time_period='augmented_val', window_size=config.window_size, train_data_ref=val_data)
        
        # Set indices
        train_indices = list(range(len(train_dataset_new)))
        valid_indices = list(range(len(val_dataset_new)))
        
        return valid_indices, train_dataset_new, val_dataset_new


# ============================================================================
# 数据集和DataLoader创建工厂
# ============================================================================

class DatasetCreator:
    """数据集创建工厂"""
    
    @staticmethod
    def create_timeseries_datasets(config):
        """创建时序数据集"""
        if config.include_labels:
            # 标准时序：包含X和y
            daydata_train = CCB_TimeSeriesDataset(config.train_time_period, config.window_size)
            train_last_window = daydata_train.data[:, -config.window_size+1:, :]
            daydata_test_original = CCB_TimeSeriesDataset(config.test_time_period, config.window_size)
            augmented_test_data = np.concatenate(
                [train_last_window, daydata_test_original.data],
                axis=1
            )
            daydata_test = CCB_TimeSeriesDataset(time_period='augmented_test', window_size=config.window_size, train_data_ref=augmented_test_data)
        else:
            # 市场时序：只有X
            daydata_train = CCB_MarketDataset(config.train_time_period, config.window_size)
            train_last_window = daydata_train.data[:, -config.window_size+1:, :]
            daydata_test_original = CCB_MarketDataset(config.test_time_period, config.window_size)
            augmented_test_data = np.concatenate(
                [train_last_window, daydata_test_original.data],
                axis=1
            )
            daydata_test = CCB_MarketDataset(time_period='augmented_test', window_size=config.window_size, train_data_ref=augmented_test_data)
        
        return daydata_train, daydata_test
    
    @staticmethod
    def create_crosssection_datasets(config):
        """创建截面数据集"""
        if config.include_labels:
            # 标准截面：包含X和y
            daydata_train = CCB_CrossSectionDataset(config.train_time_period)
            daydata_test = CCB_CrossSectionDataset(config.test_time_period)
        else:
            # 市场截面：只有X
            daydata_train = None  # 市场模式不需要训练集
            daydata_test = CCB_MKT_CrossSectionDataset(config.test_time_period)
        
        return daydata_train, daydata_test


class DataLoaderCreator:
    """DataLoader创建工厂"""
    
    @staticmethod
    def create_timeseries_collate_function(device):
        """创建时序数据的collate函数"""
        def squeeze_collate(batch):
            """当batch_size=1时自动去除批次维度"""
            if len(batch) == 1:
                # 转换numpy到Tensor + 压缩批次维度
                x, y = batch[0]
                return (
                    torch.tensor(x, dtype=torch.float32).to(device),
                    torch.tensor(y, dtype=torch.float32).to(device)
                )
            else:
                raise ValueError('batch_size can only be 1')
        return squeeze_collate
    
    @staticmethod
    def create_market_collate_function(device):
        """创建市场数据的collate函数（只有X）"""
        def market_squeeze_collate(batch):
            """当batch_size=1时自动去除批次维度，只处理X数据"""
            if len(batch) == 1:
                # 只返回X数据，转换为Tensor
                x = batch[0]
                return torch.tensor(x, dtype=torch.float32).to(device)
            else:
                raise ValueError('batch_size can only be 1')
        return market_squeeze_collate
    
    @staticmethod
    def determine_batch_sizes(config, datasets):
        """确定批次大小"""
        if isinstance(config.batchsize, int):
            return config.batchsize
        elif config.batchsize == 'all':
            # Return sizes for each dataset
            return [len(dataset) if dataset is not None else 0 for dataset in datasets]
        else:
            raise ValueError('batchsize must be an integer or string "all"')


class CCBDataLoaderFactory:
    """CCB DataLoader工厂类"""
    
    @staticmethod
    def create_timeseries_dataloaders(config):
        """创建时序DataLoader"""
        # Step 1: Create base datasets
        daydata_train, daydata_test = DatasetCreator.create_timeseries_datasets(config)
        
        if config.include_labels:
            # 标准时序：需要验证集
            # Step 2: Generate validation set based on strategy
            if config.val_sample_mode == 'random':
                valid_indices, train_override, val_override = ValidationSetGenerator.generate_random_windows(daydata_train, config)
            elif config.val_sample_mode == 'tail':
                valid_indices, train_override, val_override = ValidationSetGenerator.generate_tail_windows(daydata_train, config)
            elif config.val_sample_mode == 'random_days':
                valid_indices, train_override, val_override = ValidationSetGenerator.generate_random_days(daydata_train, config)
            
            print(f"Validation Selected windows: {valid_indices}")
            
            # Step 3: Create final datasets
            if config.val_sample_mode == 'random_days':
                # random_days mode: already separated training and validation sets
                train_dataset = train_override
                val_dataset = val_override
                test_dataset = daydata_test
            else:
                # random and tail modes: original logic
                train_indices = list(set(range(len(daydata_train))) - set(valid_indices))
                # Create subsets
                train_dataset = Subset(daydata_train, train_indices)
                val_dataset = Subset(daydata_train, valid_indices)
                test_dataset = daydata_test
            
            print(f"Train Dataset size: {len(train_dataset)}")
            print(f"Validation Dataset size: {len(val_dataset)}")
            print(f"Test Dataset size: {len(test_dataset)}")
            
            # Step 4: Create DataLoaders
            squeeze_collate = DataLoaderCreator.create_timeseries_collate_function(config.device)
            batch_sizes = DataLoaderCreator.determine_batch_sizes(config, [train_dataset, val_dataset, test_dataset])
            
            if isinstance(batch_sizes, list):
                train_batchsize, val_batchsize, test_batchsize = batch_sizes
            else:
                train_batchsize = val_batchsize = test_batchsize = batch_sizes
            
            # Handle shuffling
            train_subset_indices = list(range(len(train_dataset)))
            if config.shuffle_time:
                random.shuffle(train_subset_indices)
            train_sampler = SubsetRandomSampler(train_subset_indices) if config.shuffle_time else None
            
            # Create DataLoaders
            train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, sampler=train_sampler, collate_fn=squeeze_collate)
            val_dataloader = DataLoader(val_dataset, batch_size=val_batchsize, collate_fn=squeeze_collate)
            test_dataloader = DataLoader(test_dataset, batch_size=test_batchsize, collate_fn=squeeze_collate)
            
            # Print summary information
            CCBDataLoaderFactory._print_timeseries_summary(train_dataloader, val_dataloader, test_dataloader)
            
            return train_dataloader, val_dataloader, test_dataloader
        
        else:
            # 市场时序：只返回测试集
            test_dataset = daydata_test
            
            batch_sizes = DataLoaderCreator.determine_batch_sizes(config, [test_dataset])
            
            if isinstance(batch_sizes, list):
                test_batchsize = batch_sizes[0]
            else:
                test_batchsize = batch_sizes
            
            # Use special collate function for MKT TimeSeries data (X only)
            market_squeeze_collate = DataLoaderCreator.create_market_collate_function(config.device)
            test_dataloader = DataLoader(test_dataset, batch_size=test_batchsize, collate_fn=market_squeeze_collate)
            
            # Print summary information
            for batch_X in test_dataloader:
                test_shape = batch_X.shape
                break
            
            print(f"Market Test Dataset size: {len(test_dataset)}")
            print(f"Test batch size: {test_batchsize}")
            print(f"Test batch_X shape: {test_shape}")
            
            return test_dataloader
    
    @staticmethod
    def create_crosssection_dataloaders(config):
        """创建截面DataLoader"""
        daydata_train, daydata_test = DatasetCreator.create_crosssection_datasets(config)
        
        if config.include_labels:
            # 标准截面：返回训练集和测试集
            train_indices = list(range(len(daydata_train)))
            test_indices = list(range(len(daydata_test)))
            
            batch_sizes = DataLoaderCreator.determine_batch_sizes(config, [daydata_train, daydata_test])
            
            if isinstance(batch_sizes, list):
                train_batchsize, test_batchsize = batch_sizes
            else:
                train_batchsize = test_batchsize = batch_sizes
            
            train_sampler = SubsetRandomSampler(train_indices) if config.shuffle_time else None
            test_sampler = SubsetRandomSampler(test_indices) if config.shuffle_time else None
            
            train_dataloader = DataLoader(daydata_train, batch_size=train_batchsize, sampler=train_sampler)
            test_dataloader = DataLoader(daydata_test, batch_size=test_batchsize, sampler=test_sampler)
            
            return train_dataloader, test_dataloader
        
        else:
            # 市场截面：只返回测试集
            batch_sizes = DataLoaderCreator.determine_batch_sizes(config, [daydata_test])
            
            if isinstance(batch_sizes, list):
                test_batchsize = batch_sizes[0]
            else:
                test_batchsize = batch_sizes
            
            # Use original behavior - no special collate function for compatibility
            test_dataloader = DataLoader(daydata_test, batch_size=test_batchsize)
            
            return test_dataloader
    
    @staticmethod
    def _print_timeseries_summary(train_dataloader, val_dataloader, test_dataloader):
        """打印时序数据加载器摘要信息"""
        # Get data shapes
        for batch_X, _ in train_dataloader:
            train_shape = batch_X.shape
            break
        for batch_X, _ in val_dataloader:
            val_shape = batch_X.shape
            break
        for batch_X, _ in test_dataloader:
            test_shape = batch_X.shape
            break

        print("train_dataloader: {:d} | Validation dataloader: {:d} | Test dataloader: {:d}".format(
            len(train_dataloader), len(val_dataloader), len(test_dataloader)))
        print("train_batch_X: {} | val_batch_X: {} | test_batch_X: {}".format(
            train_shape, val_shape, test_shape))


# ============================================================================
# 主要API函数
# ============================================================================

def get_CCB_dataloader(config):
    """
    统一的CCB DataLoader入口函数
    
    Args:
        config: CCBDataLoaderConfig对象
    
    Returns:
        根据配置返回相应的DataLoader：
        - TimeSeries + Labels: train_dataloader, val_dataloader, test_dataloader
        - TimeSeries + No Labels: test_dataloader (市场预测模式)
        - CrossSection + Labels: train_dataloader, test_dataloader
        - CrossSection + No Labels: test_dataloader (市场预测模式)
    """
    
    print(f"🚀 Creating {config} DataLoader...")
    
    # 调用对应的工厂方法
    if config.data_type == 'timeseries':
        return CCBDataLoaderFactory.create_timeseries_dataloaders(config)
    elif config.data_type == 'crosssection':
        return CCBDataLoaderFactory.create_crosssection_dataloaders(config)
    else:
        raise ValueError(f"Unsupported data_type: {config.data_type}")


# ============================================================================
# 简化API - 推荐使用
# ============================================================================

def create_timeseries_dataloader(include_labels=True, window_size=30, val_sample_mode='random_days', 
                                train_period='2019-2025', test_period='2025-2026', **kwargs):
    """
    便捷函数：创建时序DataLoader
    
    Args:
        include_labels (bool): True=标准时序，False=市场预测时序
        window_size (int): 滑动窗口大小
        val_sample_mode (str): 验证集采样模式 ('random', 'tail', 'random_days')
        train_period (str): 训练时间段
        test_period (str): 测试时间段
        **kwargs: 其他配置参数
    """
    config = CCBDataLoaderConfig(
        data_type='timeseries',
        include_labels=include_labels,
        window_size=window_size,
        val_sample_mode=val_sample_mode,
        train_time_period=train_period,
        test_time_period=test_period,
        **kwargs
    )
    return get_CCB_dataloader(config)


def create_crosssection_dataloader(include_labels=True, train_period='2019-2025', 
                                  test_period='2025-2026', **kwargs):
    """
    便捷函数：创建截面DataLoader
    
    Args:
        include_labels (bool): True=标准截面，False=市场预测截面
        train_period (str): 训练时间段
        test_period (str): 测试时间段
        **kwargs: 其他配置参数
    """
    config = CCBDataLoaderConfig(
        data_type='crosssection',
        include_labels=include_labels,
        train_time_period=train_period,
        test_time_period=test_period,
        **kwargs
    )
    return get_CCB_dataloader(config)


# ============================================================================
# 向后兼容API - 保持原有函数名不变
# ============================================================================

def get_CCB_TimeSeriesDataloader(batchsize=1, train_time_period='2019-2025', test_time_period='2025-2026', 
                                shuffle_time=True, window_size=30, num_val_windows=100, val_sample_mode='random', 
                                num_val_days=100, config=None):
    """向后兼容：TimeSeries DataLoader"""
    dataloader_config = CCBDataLoaderConfig(
        data_type='timeseries',
        include_labels=True,
        batchsize=batchsize,
        train_time_period=train_time_period,
        test_time_period=test_time_period,
        shuffle_time=shuffle_time,
        window_size=window_size,
        val_sample_mode=val_sample_mode,
        num_val_windows=num_val_windows,
        num_val_days=num_val_days,
        device_config=config
    )
    return get_CCB_dataloader(dataloader_config)


def get_CCB_CrossSectionDataloader(batchsize='all', shuffle_time=False, train_time_period='2019-2025', test_time_period='2025-2026'):
    """向后兼容：CrossSection DataLoader"""
    dataloader_config = CCBDataLoaderConfig(
        data_type='crosssection',
        include_labels=True,
        batchsize=batchsize,
        shuffle_time=shuffle_time,
        train_time_period=train_time_period,
        test_time_period=test_time_period
    )
    return get_CCB_dataloader(dataloader_config)


def get_CCB_MKT_CrossSectionDataloader(batchsize='all', train_time_period='2019-2025', test_time_period='2025-2026', config=None):
    """向后兼容：MKT CrossSection DataLoader"""
    dataloader_config = CCBDataLoaderConfig(
        data_type='crosssection',
        include_labels=False,  # Market模式
        batchsize=batchsize,
        train_time_period=train_time_period,
        test_time_period=test_time_period,
        device_config=config
    )
    return get_CCB_dataloader(dataloader_config)


def get_CCB_mkt_TimeSeriesDataloader(batchsize=1, train_time_period='2019-2025', test_time_period='2025-2026', window_size=30, config=None):
    """向后兼容：MKT TimeSeries DataLoader"""
    dataloader_config = CCBDataLoaderConfig(
        data_type='timeseries',
        include_labels=False,  # Market模式
        batchsize=batchsize,
        train_time_period=train_time_period,
        test_time_period=test_time_period,
        window_size=window_size,
        device_config=config
    )
    return get_CCB_dataloader(dataloader_config)


# ============================================================================
# 测试和演示代码
# ============================================================================

if __name__ == '__main__':
    print("=== CCB DataLoader 最终统一版本测试 ===")
    
    class Config:
        def __init__(self):
            self.device = "cpu"
    
    config = Config()
    
    print("\n🔥 1. 新版简化API：")
    
    # 标准时序（包含标签）
    train_loader, val_loader, test_loader = create_timeseries_dataloader(
        include_labels=True,
        window_size=5,
        val_sample_mode='random_days',
        train_period='2019-2024',
        test_period='2024-2026',
        device_config=config
    )
    print(f"✅ 标准时序: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    # 市场时序（只有X）
    test_loader = create_timeseries_dataloader(
        include_labels=False,  # 市场模式
        window_size=5,
        train_period='2019-2024',
        test_period='2024-2026',
        device_config=config
    )
    print(f"✅ 市场时序: Test={len(test_loader)}")
    
    print("\n🔄 2. 向后兼容API测试：")
    
    # 测试原有API
    train_loader, val_loader, test_loader = get_CCB_TimeSeriesDataloader(
        batchsize=1,
        window_size=5,
        val_sample_mode='random_days',
        train_time_period='2019-2024',
        test_time_period='2024-2026',
        config=config
    )
    print(f"✅ 原版时序API: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    print("\n📊 3. API对比：")
    print("🔴 原版：4个独立函数名")
    print("🟢 新版：2个维度组合 (data_type × include_labels)")
    print("✨ 优势：更直观、更简洁、更易扩展、100%向后兼容")
    
    print("\n=== 测试完成 ===") 