import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from process_tools import *
from tqdm import tqdm

import sys
sys.path.append(
    os.path.dirname(
       os.path.dirname( __file__)
    )
)

class config:
    label_path = r'/home/hongkou/chenx/data_warehouse/labels'
    factor_path = r'/home/USB_DRIVE1/Chenx_datawarehouse/DrJin/halfday_row_factor'


market_cap = pd.read_parquet(r"/home/hongkou/chenx/data_warehouse/market_cap_2712_3883.parquet")
label = pd.read_parquet(
        os.path.join(config.label_path, f'labels.parquet')
    )

def read_factors_align_fix(file_path:str,mktcap_inner: pd.DataFrame, mktcap_outer: pd.DataFrame,label_inner: pd.DataFrame,label_outer: pd.DataFrame,sample_set:str = 'inner',inner_start:str = '2016-05-01',outer_start:str = '2021-01-01',outer_end:str = '2022-03-19'):
    f = pd.read_parquet(file_path)
    f.index = pd.to_datetime(f.index.astype(str))
    f_inner, f_outer = split_by_datetime_index(f, inner_start, outer_start, outer_end)
    if sample_set == 'inner':
        f_align_mkt = align_fix_with_market(f_inner, mktcap_inner)
        f_align_label = align_with_label(f_inner, label_inner)
        f_align_label = fillna_aligned(f_align_label, label_inner)
        return f_align_mkt, f_align_label

    elif sample_set == 'outer':
        f_align_mkt = align_fix_with_market(f_outer, mktcap_outer)
        f_align_label = align_with_label(f_outer, label_outer)
        f_align_label = fillna_aligned(f_align_label, label_outer)
        return f_align_mkt, f_align_label

file_list = [os.path.join(config.factor_path, f'half_F{i}.parquet') for i in range(1, 130)]

if __name__=='__main__':
    TIME_PERIODS = [
        {
            "outer_start": "2021-01-01",
            "outer_end": "2022-01-01",
            "path_suffix": "2021-2022"
        },
        {
            "outer_start": "2022-01-01",
            "outer_end": "2023-01-01",
            "path_suffix": "2022-2023"
        },
        {
            "outer_start": "2023-01-01",
            "outer_end": "2024-01-01",
            "path_suffix": "2023-2024"
        },
        {
            "outer_start": "2024-01-01",
            "outer_end": "2025-03-01",
            "path_suffix": "2024-2025"
        }
    ]
    INNER_START = "2016-01-05"
    BASE_SAVE_DIR = "/home/USB_DRIVE1/Chenx_datawarehouse/DrJin/factors_rolling/"
    for period in TIME_PERIODS:
        print(f"\nProcessing period: {period['path_suffix']}")
        mkt_save_dir = os.path.join(BASE_SAVE_DIR, f"r_market_align_factor/{period['path_suffix']}")
        label_factor_dir = os.path.join(BASE_SAVE_DIR, f"r_label_align_factor/{period['path_suffix']}")
        label_dir = os.path.join(BASE_SAVE_DIR, f"r_label/{period['path_suffix']}")
        os.makedirs(mkt_save_dir, exist_ok=True)
        os.makedirs(label_factor_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        mktcap_inner, mktcap_outer = split_by_datetime_index(
            market_cap,
            INNER_START,
            period["outer_start"],
            period["outer_end"]
        )
        label_inner, label_outer = split_by_datetime_index(
            label,
            INNER_START,
            period["outer_start"],
            period["outer_end"]
        )
        label_inner.to_parquet(os.path.join(label_dir, "label_inner.parquet"))
        label_outer.to_parquet(os.path.join(label_dir, "label_outer.parquet"))

        for mode in ["inner", "outer"]:
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(read_factors_align_fix,file,mktcap_inner, mktcap_outer,label_inner,label_outer,mode,INNER_START,period["outer_start"],period["outer_end"]): file for file in file_list}

                with tqdm(as_completed(futures), total=len(futures),
                          desc=f"{period['path_suffix']} {mode} Processing") as pbar:
                    for future in pbar:
                        try:
                            f_align_mkt, f_align_label = future.result()
                            file = futures[future]

                            file_tag = os.path.basename(file).split(".parquet")[0]

                            mkt_path = os.path.join(
                                mkt_save_dir,
                                f"{file_tag}_mkt_{mode}.parquet"
                            )
                            f_align_mkt.to_parquet(mkt_path)

                            label_path = os.path.join(
                                label_factor_dir,
                                f"{file_tag}_label_{mode}.parquet"
                            )
                            f_align_label.to_parquet(label_path)

                        except Exception as e:
                            print(f"\nError processing: {str(e)}")
                            break
    # inner_start = '2017-01-10'
    # outer_start = '2021-01-01'
    # outer_end = '2022-01-01'
    # mkt_factor_save_path = '/home/hongkou/chenx/data_warehouse/CY_1430_factors/r_market_align_factor/2021-2022'
    # label_factor_save_path = '/home/hongkou/chenx/data_warehouse/CY_1430_factors/r_label_align_factor/2021-2022'
    # label_save_path = '/home/hongkou/chenx/data_warehouse/CY_1430_factors/r_label/2021-2022'
    # mktcap_inner, mktcap_outer = split_by_datetime_index(market_cap, inner_start, outer_start, outer_end)
    # label_inner, label_outer = split_by_datetime_index(label, inner_start, outer_start, outer_end)
    # label_inner.to_parquet(os.path.join(label_save_path, f'label_inner.parquet'))
    # label_outer.to_parquet(os.path.join(label_save_path, f'label_outer.parquet'))
    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     futures = {
    #         executor.submit(read_factors_align_fix, file, 'inner', inner_start, outer_start, outer_end, mktcap_inner, mktcap_outer): file
    #         for file in file_list
    #     }
    #
    #     future_process = tqdm(as_completed(futures), total=len(futures),
    #                           desc="Processing days")
    #     for future in future_process:
    #         f_align_mkt, f_align_label = future.result()
    #         file = futures[future]
    #         f_align_mkt.to_parquet(os.path.join(mkt_factor_save_path, f'{file.split("factor_CY")[-1]}_mkt_align_inner.parquet'))
    #         f_align_label.to_parquet(os.path.join(label_factor_save_path, f'{file.split("factor_CY")[-1]}_label_align_inner.parquet'))
    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     futures = {
    #         executor.submit(read_factors_align_fix, file, 'outer', inner_start, outer_start, outer_end, mktcap_inner, mktcap_outer): file
    #         for file in file_list
    #     }
    #
    #     future_process = tqdm(as_completed(futures), total=len(futures),
    #                           desc="Processing days")
    #     for future in future_process:
    #         f_align_mkt, f_align_label = future.result()
    #         file = futures[future]
    #         f_align_mkt.to_parquet(os.path.join(mkt_factor_save_path, f'{file.split("factor_CY")[-1]}_mkt_align_outer.parquet'))
    #         f_align_label.to_parquet(os.path.join(label_factor_save_path, f'{file.split("factor_CY")[-1]}_lbl_align_outer.parquet'))



# for file in tqdm(file_list):
#     f_align_mkt, f_align_label = read_factors_align_fix(file,'inner',inner_start,outer_start,outer_end)
#     print("f_align_mkt shape",f_align_mkt.shape)
#     print("f_align_label shape",f_align_label.shape)
#     f_align_mkt.to_parquet(os.path.join(save_path, f'{os.path.basename(file)}_{outer_start}_{outer_end}_mkt_inner.parquet'))
#     f_align_label.to_parquet(os.path.join(save_path, f'{os.path.basename(file)}_{outer_start}_{outer_end}_label_outer.parquet'))