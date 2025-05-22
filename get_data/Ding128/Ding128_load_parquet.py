import os
from concurrent import futures

import numpy as np
import pandas as pd
import glob

from config import get_config
config = get_config()


def load_Ding128_x_parquet(x_type='factor', sample_set='inner', align_type='alb'):
    assert sample_set == 'inner' or sample_set == 'outer'

    if x_type == 'factor':
        file_list = [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                  f'{sample_set}_d00{i}_{align_type}.parquet') for i in range(1, 10)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_d0{i}_{align_type}.parquet') for i in range(10, 15)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_d0{i}_{align_type}.parquet') for i in range(16, 18)]
        # file_list = file_list + [os.path.join(config.factor_path, f'{i}_factor_{sample_set}_aligned.parquet') for i in range(146,150)]

        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(15, 16)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(18, 22)]

        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(24, 34)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(35, 36)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(46, 47)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(48, 50)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(52, 55)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(57, 61)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(65, 69)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(78, 80)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(83, 84)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in [87, 93, 94]]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m0{i}_{align_type}.parquet') for i in range(98, 99)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m{i}_{align_type}.parquet') for i in range(102, 103)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m{i}_{align_type}.parquet') for i in range(125, 127)]

        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m{i}_{align_type}.parquet') for i in range(168, 169)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m{i}_{align_type}.parquet') for i in range(169, 170)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m{i}_{align_type}.parquet') for i in range(172, 173)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m{i}_{align_type}.parquet') for i in range(173, 174)]

        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m{i}_{align_type}.parquet') for i in range(184, 185)]

    elif x_type == 'feature_statistics':

        file_list = [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                  f'{sample_set}_m000{i}_{align_type}.parquet') for i in range(1, 10)]
        file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                              f'{sample_set}_m00{i}_{align_type}.parquet') for i in range(10, 13)]
    else:
        raise ValueError('x_type 请输入："factor" 或 "feature_statistics"')

    with futures.ThreadPoolExecutor(max_workers=24) as thread_pool:
        futures_list = []
        for file in file_list:
            futures_list.append(thread_pool.submit(lambda file: pd.read_parquet(file), file))

    def standard(df):
        return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

    return np.array([standard(future.result()).values for future in futures_list]).transpose(2, 1, 0)

def load_Ding128_tbtstats_pqt(sample_set='inner', align_type='alb'):
    assert sample_set == 'inner' or sample_set == 'outer'

    file_list = [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                              f'{sample_set}_t000{num}_{align_type}.parquet') for num in range(5, 10)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t00{num}_{align_type}.parquet') for num in range(10, 14)]

    with futures.ThreadPoolExecutor(max_workers=24) as thread_pool:
        futures_list = []
        for file in file_list:
            futures_list.append(thread_pool.submit(lambda file: pd.read_parquet(file), file))

    # t = [future.result().values for future in futures_list]
    # for ti in t:
    #     print(ti.shape)
    def standard(df):
        return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

    return np.array([standard(future.result()).values for future in futures_list]).transpose(2, 1, 0)


def load_Ding128_tbtfactors_pqt(sample_set='inner', align_type='alb'):
    assert sample_set == 'inner' or sample_set == 'outer'

    file_list = [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                              f'{sample_set}_t000{num}_{align_type}.parquet') for num in range(1, 5)]
    # file_list = []
    # file_list = [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'), f'{sample_set}_t00{num}_{align_type}.parquet') for num in range(1, 5)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t00{num}_{align_type}.parquet') for num in range(5, 6)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t00{num}_{align_type}.parquet') for num in range(6, 10)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(16, 21)]

    # file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'), f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(15, 16)]

    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(24, 26)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(40, 41)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(36, 37)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(39, 40)]

    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(33, 34)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(35, 36)]

    # +t52
    # file_list = file_list + [os.path.join(eval(f'config.Ding128_alb_{sample_set}_path'), f'{sample_set}_t0{num}_alb.parquet') for num in range(52, 53)]
    # file_list = file_list + [os.path.join(eval(f'config.Ding128_alb_{sample_set}_path'), f'{sample_set}_t0{num}_alb.parquet') for num in range(85, 86)]

    # file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'), f'{sample_set}_t0{num}_{align_type}.parquet') for num in [53, 55, 56, 57, 58]]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t0{num}_{align_type}.parquet') for num in
                             [53, 55, 56, 57, 58, 59, 60, 61]]
    # file_list = file_list + [os.path.join(eval(f'config.Ding128_alb_{sample_set}_path'), f'{sample_set}_t0{num}_alb.parquet') for num in range(57, 58)]
    # file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'), f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(92, 98)] # 7.37, 5.01
    # file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'), f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(98, 100)] #98-103 7.35, 5.05
    # file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'), f'{sample_set}_t{num}_{align_type}.parquet') for num in range(100, 104)]

    # file_list = file_list + [os.path.join(eval(f'config.Ding128_alb_{sample_set}_path'), f'{sample_set}_t00{num}_alb.parquet') for num in range(23, 24)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t00{num}_{align_type}.parquet') for num in range(24, 27)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t00{num}_{align_type}.parquet') for num in range(28, 32)]
    file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'),
                                          f'{sample_set}_t00{num}_{align_type}.parquet') for num in range(14, 24)]

    # file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'), f'{sample_set}_t00{num}_{align_type}.parquet') for num in range(24, 33)]
    # file_list = file_list + [os.path.join(config.tbtDing128_path_aggday, f'{sample_set}_t0{num}_aligned.parquet') for num in range(68, 71)]
    # file_list = file_list + [os.path.join(config.tbtDing128_path_aggday, f'{sample_set}_t0{num}_aligned.parquet') for num in range(62, 93)]
    # file_list = file_list + [os.path.join(config.tbtDing128_path_aggday, f'{sample_set}_t0{num}_aligned.parquet') for num in range(79, 80)]
    # file_list = file_list + [os.path.join(eval(f'config.Ding128_{align_type}_{sample_set}_path'), f'{sample_set}_t0{num}_{align_type}.parquet') for num in range(48, 52)]

    def standard(df):
        return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

    with futures.ThreadPoolExecutor(max_workers=24) as thread_pool:
        futures_list = []
        for file in file_list:
            futures_list.append(thread_pool.submit(lambda file: pd.read_parquet(file), file))

    # t = [future.result().values for future in futures_list]
    # for ti in t:
    #     print(ti.shape)
    def standard(df):
        return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

    return np.array([standard(future.result()).values for future in futures_list]).transpose(2, 1, 0)


def load_Ding128_y_parquet(sample_set='inner'):
    assert sample_set == 'inner' or sample_set == 'outer'

    def standard(df):
        return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)
    return np.expand_dims(
        standard(pd.read_parquet(
            os.path.join(config.Ding128_label_path, f'label_{sample_set}.parquet')
        )).values.transpose(1, 0), axis=2)
