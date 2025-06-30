import os
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import sys
import re

# 添加父目录到路径以导入get_data_from_DB模块
sys.path.append(str(Path(__file__).parent.parent))
from get_data_from_DB import get_bond_stk_name_df

input_root = Path(r'D:\raw_data\stock_min_data_per_day')
output_root = Path(r"D:\chenxing\Finforecast\factor_warehouse\one_minute_parquet")


def process_subfolder(subfolder: Path, stock_bond_df: pd.DataFrame):
    """处理单个子文件夹的数据"""
    try:
        # 1. 读并拼接
        dfs = []
        
        for file in subfolder.iterdir():
            if not file.is_file():
                continue
            df = pd.read_pickle(file)
            assert df.shape == (240, 59)
            dfs.append(df)
                
        assert len(dfs) > 0

        # 使用更高效的方式合并数据
        combined = pd.concat(dfs, ignore_index=True)
        cols = [c for c in combined.columns if c not in ('code','datetime')]

        # 2. 输出目录
        out_sub = output_root / subfolder.name
        ccb = pd.read_parquet(out_sub / 'close.parquet')
        ccb_cols = ccb.columns.tolist()

        codes_map = stock_bond_df[stock_bond_df['bond_code'].isin(ccb_cols)].copy()
        ccb_missing = set(ccb_cols) - set(stock_bond_df['bond_code'])
        if ccb_missing:
            print(f"缺失的 bond_code 有 {len(ccb_missing)} 个，{list(ccb_missing)}")
        
        valid_data = combined[combined['code'].isin(codes_map['stock_code'].tolist())]
        
        filtered_count = len(combined) - len(valid_data)

        assert filtered_count > 0

        # 4. pivot & 写 Parquet（带进度条）
        for col in cols:
            # 检查重复数据
            pivot_subset = valid_data[['datetime', 'code', col]]
            duplicates = pivot_subset.duplicated(subset=['datetime', 'code'], keep=False)
            
            if duplicates.any():
                print(f"发现重复数据：")
                print(pivot_subset[duplicates].head(10))
                # 对重复数据取平均值
                pivot_data = pivot_subset.groupby(['datetime', 'code'])[col].last().reset_index()
            else:
                pivot_data = pivot_subset.copy()
            wide = pivot_data.pivot(index='datetime', columns='code', values=col)
            stock_list = codes_map['stock_code'].tolist()
            wide_sel   = wide.reindex(columns=stock_list)
            wide_sel.columns = codes_map['bond_code'].tolist()
            wide_sel.columns = wide_sel.columns.astype('int32')
            wide_sel.columns.name = 'code'
            
            assert wide_sel.shape[0] == 240, f"{subfolder.name} {col} 数据长度不等于240"
            assert wide_sel.shape[1] == len(codes_map), f"{subfolder.name} {col} 数据列数不等于存在的债券"
            
            # 保存为parquet文件，文件名加上stock_前缀
            wide_sel.to_parquet(out_sub / f"stock_{col}.parquet")


    except Exception as e:
        raise ValueError(f"处理文件夹 {subfolder.name} 时发生错误: {e}")

def process_subfolder_wrapper(args):
    """包装函数，用于多进程处理"""
    subfolder, stock_bond_df = args
    return process_subfolder(subfolder, stock_bond_df)

if __name__ == '__main__':
        
    print("正在获取股票-转债映射关系...")
    stock_bond_df = get_bond_stk_name_df()
    
    # 收集所有待处理子目录
    subfolders = [sd for sd in input_root.iterdir() if sd.is_dir()]
    print(f"找到 {len(subfolders)} 个子目录待处理")

    # 准备多进程参数
    process_args = [(subfolder, stock_bond_df) for subfolder in subfolders]

    # 并行执行并显示总体进度
    with ProcessPoolExecutor() as exe:
        # exe.map 会按顺序返回结果，我们用 tqdm 包裹它
        for codes_map in tqdm(exe.map(process_subfolder_wrapper, process_args),
                           total=len(subfolders),
                           desc="Processing stock data"):
            pass
            
    print("所有处理完成") 