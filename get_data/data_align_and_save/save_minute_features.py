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
    factor_path = r'/home/USB_DRIVE1/Chenx_datawarehouse/5min_factors/5min_features_ori'


market_cap = pd.read_parquet('/home/hongkou/chenx/data_warehouse/marketcap.parquet')
label = pd.read_parquet(
        os.path.join(config.label_path, f'labels.parquet')
    )

def read_factors_align_fix(file_path:str,mktcap_inner: pd.DataFrame, mktcap_outer: pd.DataFrame,label_inner: pd.DataFrame,label_outer: pd.DataFrame,sample_set:str = 'inner',inner_start:str = '2016-05-01',outer_start:str = '2021-01-01',outer_end:str = '2022-03-19'):
    f = pd.read_parquet(file_path)
    f_inner, f_outer = split_by_datetime_index(f, inner_start, outer_start, outer_end)
    if sample_set == 'inner':
        f_align_mkt = minute_align_fix_with_market(f_inner, mktcap_inner)
        # f_align_label = align_with_label(f_inner, label_inner)
        # f_align_label = fillna_aligned(f_align_label, label_inner)
        return f_align_mkt
    elif sample_set == 'outer':
        f_align_mkt = minute_align_fix_with_market(f_outer, mktcap_outer)
        # f_align_label = align_with_label(f_align_mkt, label_outer)
        # f_align_label = fillna_aligned(f_align_label, label_outer)
        return f_align_mkt

file_list = [os.path.join(config.factor_path, f'F{i}') for i in range(1, 313)]


if __name__=='__main__':
    TIME_PERIODS = [
        {
            "outer_start": "2021-01-01",
            "outer_end": "2022-03-19",
            "path_suffix": "2021-2022"
        },
        # {
        #     "outer_start": "2022-01-01",
        #     "outer_end": "2023-01-01",
        #     "path_suffix": "2022-2023"
        # },
        # {
        #     "outer_start": "2023-01-01",
        #     "outer_end": "2024-06-20",
        #     "path_suffix": "2023-2024"
        # }
    ]
    INNER_START = "2016-05-13"
    BASE_SAVE_DIR = "/home/USB_DRIVE1/Chenx_datawarehouse/5min_factors"
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
                            f_align_mkt = future.result()
                            file = futures[future]

                            file_tag = os.path.basename(file)

                            mkt_path = os.path.join(
                                mkt_save_dir,
                                f"{file_tag}_mkt_{mode}.parquet"
                            )
                            f_align_mkt.to_parquet(mkt_path)

                            # label_path = os.path.join(
                            #     label_factor_dir,
                            #     f"{file_tag}_label_{mode}.parquet"
                            # )
                            # f_align_label.to_parquet(label_path)

                        except Exception as e:
                            print(f"\nError processing {file}: {str(e)}")
                            break
