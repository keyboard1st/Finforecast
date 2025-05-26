import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(
    os.path.dirname(
       os.path.dirname( __file__)
    )
)

from process_tools import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)



class config:
    label_path = r'/home/hongkou/chenx/data_warehouse/labels'
    factor_path = r'/home/USB_DRIVE1/gene_10min_npy'

index = np.load('/home/hongkou/index_tradingday_201420250228.npy')
ticker = np.load('/home/hongkou/index_Ticker_201420200604.npy')
f = np.load('/home/USB_DRIVE1/gene_10min_npy/abn_tvr.npy')
T, N, Bars_oneday = f.shape
index_days = index[:T]
index_minutes = np.repeat(index_days, Bars_oneday)
market_cap = pd.read_parquet('/home/hongkou/chenx/data_warehouse/market_cap_2712_3883.parquet')
label = pd.read_parquet(
        os.path.join(config.label_path, f'labels.parquet')
    )

def read_factors_align_fix(file_path,mkt_inner, mkt_outer,sample_set,inner_start,outer_start,outer_end):
    factor = np.load(file_path)
    t1_minute = factor.transpose(0, 2, 1).reshape(T * Bars_oneday, N)
    t1_df = pd.DataFrame(t1_minute)
    t1_df.index = index_minutes
    t1_df.columns = ticker
    t1_df_shift = t1_df.shift(3)
    t1_df_shift.to_parquet(f'/home/USB_DRIVE1/Chenx_datawarehouse/10min_factors/factor_row/{os.path.basename(file_path)}_row.parquet')
    f_inner, f_outer = split_by_datetime_index(t1_df_shift, inner_start, outer_start, outer_end)
    if sample_set == 'inner':
        f_align_mkt_inner = minute_align_fix_with_market(f_inner, mkt_inner)
        return f_align_mkt_inner
    elif sample_set == 'outer':
        f_align_mkt_outer = minute_align_fix_with_market(f_outer, mkt_outer)
        return f_align_mkt_outer

exclude = {'ret30min.npy'}
file_list = sorted(os.path.join(config.factor_path, fn) for fn in os.listdir(config.factor_path) if fn.endswith('.npy') and fn not in exclude)

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
        }
    ]
    INNER_START = "2016-01-05"
    BASE_SAVE_DIR = "/home/USB_DRIVE1/Chenx_datawarehouse/10min_factors"
    for period in TIME_PERIODS:
        print(f"\nProcessing period: {period['path_suffix']}")
        mkt_save_dir = os.path.join(BASE_SAVE_DIR, f"r_market_align_factor/{period['path_suffix']}")
        label_dir = os.path.join(BASE_SAVE_DIR, f"r_label/{period['path_suffix']}")
        os.makedirs(mkt_save_dir, exist_ok=True)
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
                futures = {executor.submit(read_factors_align_fix,file,mktcap_inner, mktcap_outer,mode,INNER_START,period["outer_start"],period["outer_end"]): file for file in file_list}

                with tqdm(as_completed(futures), total=len(futures),
                          desc=f"{period['path_suffix']} {mode} Processing") as pbar:
                    for future in pbar:
                        try:
                            f_align_mkt = future.result()
                            file = futures[future]

                            i = os.path.basename(file).split(".npy")[0]

                            mkt_path = os.path.join(
                                mkt_save_dir,
                                f"{i}_mkt_{mode}.parquet"
                            )
                            f_align_mkt.to_parquet(mkt_path)

                        except Exception as e:
                            print(f"\nError processing {file}: {str(e)}")
                            continue
