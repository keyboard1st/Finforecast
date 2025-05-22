import os
from concurrent import futures

import numpy as np
import pandas as pd

from config import get_config
config = get_config()

def load_DrJin129_x_parquet(sample_set='inner'):
    assert sample_set == 'inner' or sample_set == 'outer'
    halfday_list = sorted(
        [os.path.join(config.DrJin129_hal_factor_path, f"{i}") for i in os.listdir(config.DrJin129_hal_factor_path) if
         i.split('_')[-2] == sample_set])
    allday_list = sorted(
        [os.path.join(config.DrJin129_all_factor_path, f"{i}") for i in os.listdir(config.DrJin129_all_factor_path) if
         i.split('_')[-2] == sample_set])
    with futures.ThreadPoolExecutor(max_workers=24) as executor:
        halfday_dfs = list(executor.map(pd.read_parquet, halfday_list))  # 直接获取 DataFrame 列表
    with futures.ThreadPoolExecutor(max_workers=24) as executor:
        allday_dfs = list(executor.map(pd.read_parquet, allday_list))

    first_file = pd.read_parquet(halfday_list[50])
    pd.testing.assert_frame_equal(first_file, halfday_dfs[50])
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

def load_DrJin129_y_parquet(sample_set='inner'):
    assert sample_set == 'inner' or sample_set == 'outer'

    def standard(df):
        return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

    return np.expand_dims(
        standard(pd.read_parquet(
            os.path.join(config.DrJin_label_path, f'label_{sample_set}.parquet')
        )).values.transpose(1, 0), axis=2)


    # return np.expand_dims(
    #     pd.read_parquet(
    #     os.path.join(config.label_path, f'label_{sample_set}.parquet')
    # ).values.transpose(1, 0), axis=2)
    # return standard()