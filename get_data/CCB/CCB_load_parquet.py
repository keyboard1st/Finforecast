import os
from concurrent import futures

import numpy as np
import pandas as pd
import glob

import sys
if any('ipykernel_launcher' in a for a in sys.argv):
    sys.argv = sys.argv[:1]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    os.pardir,
                                    os.pardir))
sys.path.insert(0, ROOT)
from config import get_config
config = get_config()

exclude = {1, 3, 5, 6}  #剔除的F因子



def standard(df):
    return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

def load_x_CCB_parquet(time_period='2019-2021'):
    start_str, end_str = time_period.split('-')
    start, end = int(start_str), int(end_str)
    years = list(range(start, end))
    time_periods = [f"{y}-{y+1}" for y in years]

    x_list = []
    for period in time_periods:
        file_list = []
        file_list += [os.path.join(config.CCB_factor_path, period, f'F{i}.parquet') for i in range(1, 34) if i not in exclude]
        file_list += [os.path.join(config.CCB_factor_path, period, f'factor{i}.parquet') for i in range(1, 84)]
        file_list = sorted(file_list)

        with futures.ThreadPoolExecutor(max_workers=24) as executor:
            dfs = list(executor.map(pd.read_parquet, file_list))

        arrays = [standard(df).values for df in dfs]
        # Stack arrays along new factor axis: shape (num_factors, days, stocks)
        stacked = np.stack(arrays, axis=0)
        # Transpose to (days, stocks, num_factors)
        cur = stacked.transpose(1, 2, 0) 

        x_list.append(cur)

    # Concatenate all periods along the day axis (axis=0)
    x = np.concatenate(x_list, axis=0)
    # (B, T, C)
    return x.transpose(1, 0, 2) 



def load_y_CCB_parquet(time_period='2019-2021'):
    start_str, end_str = time_period.split('-')
    start, end = int(start_str), int(end_str)
    years = list(range(start, end))
    time_periods = [f"{y}-{y+1}" for y in years]
    y_list = []
    for period in time_periods:
        y_list.append(standard(pd.read_parquet(
            os.path.join(config.CCB_label_path, period, f'label.parquet')
        )).values)
    y = np.concatenate(y_list, axis=0).transpose(1, 0)
    # (B, T, 1)
    return np.expand_dims(y, axis=2)

if __name__ == '__main__':
    
    x = load_x_CCB_parquet()
    print(x.shape)
    y = load_y_CCB_parquet()
    print(y.shape)
