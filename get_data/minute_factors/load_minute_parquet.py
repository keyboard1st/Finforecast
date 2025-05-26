import os
from concurrent import futures

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def standard(df):
    return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

def load_10minute_rolling_x_parquet(x_type = 'lbl_align',sample_set = 'inner', time_period:str = '2021-2022'):
    from config import get_config
    config = get_config()
    label = pd.read_parquet(os.path.join(config.min10_rolling_label_path, f'{time_period}/label_{sample_set}.parquet'))
    target_cols = label.columns
    file_list = [os.path.join(config.min10_rolling_factor_path, f'{time_period}/F{i}_{sample_set}.parquet') for i in range(1, 38)]

    def _load_std_and_select(path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)  # 读全表
        df = standard(df)  # 行标准化
        if x_type == 'lbl_align':
            # 仅在 lbl_align 模式下做列对齐
            df = df.loc[:, target_cols]
        return df

    with futures.ThreadPoolExecutor(max_workers=24) as executor:
        processed = list(executor.map(_load_std_and_select, file_list))
    factor_arr = np.array([df.values for df in processed]).transpose(2, 1, 0)
    return factor_arr

def load_10minute_rolling_y_parquet(sample_set = 'inner', time_period:str = '2021-2022'):
    from config import get_config
    config = get_config()

    return np.expand_dims(standard(pd.read_parquet(os.path.join(config.min10_rolling_label_path, f'{time_period}/label_{sample_set}.parquet'))).values.transpose(1, 0), axis=2)


if __name__ == '__main__':
    x = load_10minute_rolling_x_parquet(x_type = 'lbl_align',sample_set = 'inner', time_period = '2021-2022')

    print(x.shape)

    y = load_10minute_rolling_y_parquet(sample_set = 'inner', time_period = '2021-2022')

    print(y.shape)
