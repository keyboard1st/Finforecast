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
from utils.time_parser import parse_period_to_months
from config import get_config
config = get_config()

exclude = {1, 3, 5, 6}  #剔除的F因子



def standard(df):
    return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

def load_x_CCB_parquet(time_period='202107-202206'):
    
    time_periods = parse_period_to_months(time_period)

    x_list = []
    for period in time_periods:
        file_list = []
        file_list += [os.path.join(config.CCB_factor_path, period, f'F{i}.parquet') for i in range(1, 55) if i not in exclude]
        file_list += [os.path.join(config.CCB_factor_path, period, f'factor{i}.parquet') for i in range(1, 84)]
        # file_list += [os.path.join(config.CCB_factor_path, period, f'd{i}.parquet') for i in range(1, 133)]
        file_list = sorted(file_list)

        with futures.ThreadPoolExecutor(max_workers=24) as executor:
            dfs = list(executor.map(pd.read_parquet, file_list))

        arrays = [standard(df).values for df in dfs]
        # Stack arrays along new factor axis: shape (days, stocks, num_factors)
        cur = np.stack(arrays, axis=-1)
        x_list.append(cur)

    # Concatenate all periods along the day axis (axis=0)
    x = np.concatenate(x_list, axis=0)
    # (B, T, C)
    return x.transpose(1, 0, 2) 



def load_y_CCB_parquet(time_period='202107-202206'):
    time_periods = parse_period_to_months(time_period)
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
