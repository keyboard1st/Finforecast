import os
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

input_root = Path(r'D:\raw_data\ccb_min_data_per_day')
output_root = Path(r'D:\chenxing\factor_warehouse\one_minute_parquet')
output_root.mkdir(parents=True, exist_ok=True)

def process_subfolder(subfolder: Path):
    # 1. 读并拼接
    dfs = []
    for file in subfolder.iterdir():
        if not file.is_file():
            continue
        df = pd.read_pickle(file)
        assert df.shape == (240,59)
        dfs.append(df)
    if not dfs:
        return f"{subfolder.name}: empty"

    combined = pd.concat(dfs, ignore_index=True)
    cols = [c for c in combined.columns if c not in ('code','datetime')]

    # 2. 输出目录
    out_sub = output_root / subfolder.name
    out_sub.mkdir(parents=True, exist_ok=True)

    # 3. pivot & 写 Parquet（带进度条）
    for col in cols:
        wide = combined.pivot(index='datetime', columns='code', values=col)
        wide.to_parquet(out_sub / f"{col}.parquet")

    return f"{subfolder.name}: OK ({len(cols)} files)"

if __name__ == '__main__':

    # 收集所有待处理子目录
    subfolders = [sd for sd in input_root.iterdir() if sd.is_dir()]

    # 并行执行并显示总体进度
    with ProcessPoolExecutor() as exe:
        # exe.map 会按顺序返回结果，我们用 tqdm 包裹它
        for result in tqdm(exe.map(process_subfolder, subfolders),
                           total=len(subfolders),
                           desc="Processing dates"):
            print(result)
