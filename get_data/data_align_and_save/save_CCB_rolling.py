import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from process_tools import split_by_int_index, adjust_row
from tqdm import tqdm


class config:
    label_path = r'D:\chenxing\Finforecast\factor_warehouse\label_raw\label_vwap_log_cliped_filtered.parquet'
    factor_path1 = r'D:\chenxing\Finforecast\factor_warehouse\factors_pricing_raw'
    factor_path2 = r'D:\chenxing\Finforecast\factor_warehouse\factors_raw'

label = pd.read_parquet(config.label_path)

def read_factors_align_fix(file_path:str,label:pd.DataFrame,start:int,end:int):
    f = pd.read_parquet(file_path)
    f.index = f.index.astype(np.int32)
    f_re = f.reindex_like(label)
    f_split = split_by_int_index(f_re, start, end)
    f_split_clip = f_split.apply(adjust_row, axis=1)
    return f_split_clip

file_list = [os.path.join(config.factor_path1, f'factor{i}.parquet') for i in range(1, 84)] + [os.path.join(config.factor_path2, f'F{i}.parquet') for i in range(1, 34)]

if __name__=='__main__':
    TIME_PERIODS = [
        {
            "start": 20190101,
            "end": 20200101,
            "path_suffix": "2019-2020"
        },
        {
            "start": 20200101,
            "end": 20210101,
            "path_suffix": "2020-2021"
        },
        {
            "start": 20210101,
            "end": 20220101,
            "path_suffix": "2021-2022"
        },
        {
            "start": 20220101,
            "end": 20230101,
            "path_suffix": "2022-2023"
        },
        {
            "start": 20230101,
            "end": 20240101,
            "path_suffix": "2023-2024"
        },
        {
            "start": 20240101,
            "end": 20250101,
            "path_suffix": "2024-2025"
        },
        {
            "start": 20250101,
            "end": 20260101,
            "path_suffix": "2025-2026"
        },
    ]
    BASE_SAVE_DIR = r"D:\chenxing\Finforecast\factor_warehouse\factor_aligned"
    for period in TIME_PERIODS:
        print(f"\nProcessing period: {period['path_suffix']}")
        factor_dir = os.path.join(BASE_SAVE_DIR, f"r_factor/{period['path_suffix']}")
        label_dir = os.path.join(BASE_SAVE_DIR, f"r_label/{period['path_suffix']}")
        os.makedirs(factor_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        label_split = split_by_int_index(label,period["start"],period["end"])
        label_split.to_parquet(os.path.join(label_dir, "label.parquet"))

        with ProcessPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(read_factors_align_fix,file,label,period["start"],period["end"]): file for file in file_list}

            with tqdm(as_completed(futures), total=len(futures),
                        desc=f"{period['path_suffix']} Processing") as pbar:
                for future in pbar:
                    try:
                        f_align = future.result()
                        file = futures[future]

                        file_tag = os.path.basename(file).split(".parquet")[0]

                        factor_path = os.path.join(
                            factor_dir,
                            f"{file_tag}.parquet"
                        )
                        f_align.to_parquet(factor_path)

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