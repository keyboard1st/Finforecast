import os
from pathlib import Path
import pandas as pd

input_root = Path(r'D:\raw_data\stock_min_data_per_day')

def inspect_stock_codes():
    """检查原始股票数据中的code字段格式"""
    # 随机选择一个日期文件夹
    sample_date = next(d for d in input_root.iterdir() if d.is_dir())
    print(f"检查日期文件夹: {sample_date.name}")
    
    # 读取该文件夹下的所有pickle文件
    unique_codes = set()
    code_examples = []
    total_files = 0
    
    for file in sample_date.iterdir():
        if not file.is_file():
            continue
        try:
            df = pd.read_pickle(file)
            code = df['code'].iloc[0]  # 每个文件的code都是相同的
            unique_codes.add(str(code))
            if len(code_examples) < 5:
                code_examples.append((file.name, str(code)))
            total_files += 1
        except Exception as e:
            print(f"读取文件 {file} 失败: {e}")
            continue

    print(f"\n总共读取了 {total_files} 个文件")
    print(f"发现 {len(unique_codes)} 个不同的股票代码")
    print("\n示例文件及其code值:")
    for filename, code in code_examples:
        print(f"文件: {filename:<30} code: {code}")
    
    print("\n所有code值的长度统计:")
    length_stats = {}
    for code in unique_codes:
        length = len(str(code))
        length_stats[length] = length_stats.get(length, 0) + 1
    
    for length, count in sorted(length_stats.items()):
        print(f"长度为{length}的code: {count}个")
        # 显示每种长度的一些示例
        examples = [code for code in unique_codes if len(str(code)) == length][:5]
        print(f"示例: {examples}")

if __name__ == '__main__':
    inspect_stock_codes() 