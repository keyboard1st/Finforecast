import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from process_tools import *

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class config:
    label_path = r'/home/hongkou/chenx/data_warehouse/labels'
    factor_path = r'/home/USB_DRIVE1/gene_10min_npy'

index = np.load('/home/hongkou/index_tradingday_201420250228.npy')
ticker = np.load('/home/hongkou/index_Ticker_201420200604.npy')
f_sample = np.load('/home/USB_DRIVE1/gene_10min_npy/abn_tvr.npy')
T, N, Bars_oneday = f_sample.shape
index_days = index[:T]
index_minutes = np.repeat(index_days, Bars_oneday)
market_cap = pd.read_parquet('/home/hongkou/chenx/data_warehouse/market_cap_2712_3883.parquet')
label = pd.read_parquet(os.path.join(config.label_path, 'labels.parquet'))

def load_factor(file_path):
    """多进程池中执行：只做 np.load"""
    return np.load(file_path), file_path

def compute_and_save(task):
    """
    多线程池中执行：接收 (factor_array, file_path, mkt_inner, mkt_outer, sample_set, inner_start, outer_start, outer_end)
    完成所有 DataFrame 重组/shift/分割/对齐/行存盘 + 最终 parquet 存盘
    """
    factor_array, file_path, mkt_inner, mkt_outer, sample_set, inner_start, outer_start, outer_end, save_dir = task

    # —— 与原来 read_factors_align_fix 体完全一致 —— #
    # 转置、reshape
    t1_minute = factor_array.transpose(0, 2, 1).reshape(T * Bars_oneday, N)
    t1_df = pd.DataFrame(t1_minute, index=index_minutes, columns=ticker)

    # shift + 行存盘
    t1_df_shift = t1_df.shift(3)
    row_path = os.path.join(
        '/home/USB_DRIVE1/Chenx_datawarehouse/10min_factors/factor_row',
        os.path.basename(file_path) + '_row.parquet'
    )
    t1_df_shift.to_parquet(row_path)

    # 分割 & 对齐
    f_inner, f_outer = split_by_datetime_index(
        t1_df_shift, inner_start, outer_start, outer_end
    )
    if sample_set == 'inner':
        df_align = minute_align_fix_with_market(f_inner, mkt_inner)
    else:
        df_align = minute_align_fix_with_market(f_outer, mkt_outer)

    # 最终存盘
    idx = os.path.basename(file_path).split('.npy')[0]
    out_path = os.path.join(save_dir, f'{idx}_mkt_{sample_set}.parquet')
    df_align.to_parquet(out_path)


def main():
    TIME_PERIODS = [
        {"outer_start": "2021-01-01", "outer_end": "2022-01-01", "path_suffix": "2021-2022"},
        {"outer_start": "2022-01-01", "outer_end": "2023-01-01", "path_suffix": "2022-2023"},
        {"outer_start": "2023-01-01", "outer_end": "2024-01-01", "path_suffix": "2023-2024"},
    ]
    INNER_START = "2016-01-05"
    BASE_SAVE_DIR = "/home/USB_DRIVE1/Chenx_datawarehouse/10min_factors"
    file_list = [os.path.join(config.factor_path, f) for f in sorted(os.listdir(config.factor_path))]

    # 1) 永久开启一个进程池，只做 I/O
    with ProcessPoolExecutor(max_workers=8) as io_pool:
        for period in TIME_PERIODS:
            print(f"\nProcessing period: {period['path_suffix']}")
            mkt_save_dir = os.path.join(BASE_SAVE_DIR, f"r_market_align_factor/{period['path_suffix']}")
            label_dir = os.path.join(BASE_SAVE_DIR, f"r_label/{period['path_suffix']}")
            os.makedirs(mkt_save_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            # 拆市场和 label
            mktcap_inner, mktcap_outer = split_by_datetime_index(
                market_cap, INNER_START, period["outer_start"], period["outer_end"]
            )
            label_inner, label_outer = split_by_datetime_index(
                label, INNER_START, period["outer_start"], period["outer_end"]
            )
            label_inner.to_parquet(os.path.join(label_dir, "label_inner.parquet"))
            label_outer.to_parquet(os.path.join(label_dir, "label_outer.parquet"))

            for mode in ["inner", "outer"]:
                # 2) 针对本 period+mode 开一个线程池，只做计算
                with ThreadPoolExecutor(max_workers=24) as cpu_pool:
                    # 提交所有 load 任务
                    io_futs = {io_pool.submit(load_factor, fp): fp for fp in file_list}
                    # 收集并链到计算
                    cpu_futs = {}
                    for io_fut in tqdm(as_completed(io_futs)):
                        fp = io_futs[io_fut]
                        try:
                            arr, loaded_fp = io_fut.result()
                            # 打包参数提交到线程池
                            task = (
                                arr, loaded_fp,
                                mktcap_inner, mktcap_outer,
                                mode, INNER_START,
                                period["outer_start"], period["outer_end"],
                                mkt_save_dir
                            )
                            cpu_futs[cpu_pool.submit(compute_and_save, task)] = fp

                        except Exception as e:
                            print(f"\n[I/O ERROR] {fp}: {e}")

                    # 等待所有计算完成（内部已经保存好了 parquet）
                    for cpu_fut in tqdm(as_completed(cpu_futs),
                                        total=len(cpu_futs),
                                        desc=f"{period['path_suffix']} {mode}"):
                        fp = cpu_futs[cpu_fut]
                        try:
                            cpu_fut.result()
                        except Exception as e:
                            print(f"\n[COMPUTE ERROR] {fp}: {e}")
    print("All done.")


if __name__=='__main__':
    main()
