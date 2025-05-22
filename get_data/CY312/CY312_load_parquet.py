import os
from concurrent import futures

import numpy as np
import pandas as pd
import glob
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config
config = get_config()

def standard(df):
    return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)
def load_x_CY312_parquet(sample_set='inner'):
    assert sample_set == 'inner' or sample_set == 'outer'
    file_list = [os.path.join(config.CY312_factor_path, f'{sample_set}_F{i}_aligned.parquet') for i in range(1, 313)]
    with futures.ThreadPoolExecutor(max_workers=24) as thread_pool:
        futures_list = []
        for file in file_list:
            futures_list.append(thread_pool.submit(lambda file: pd.read_parquet(file), file))

    return np.array([standard(future.result()).values for future in futures_list]).transpose(2, 1, 0)

def load_y_CY312_parquet(sample_set='inner'):
    assert sample_set == 'inner' or sample_set == 'outer'

    return np.expand_dims(
        standard(pd.read_parquet(
            os.path.join(config.Ding128label_path, f'label_{sample_set}.parquet')
        )).values.transpose(1, 0), axis=2)
