import sys
import os
from pathlib import Path
import pandas as pd
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from factor_warehouse.get_data_from_DB import get_bond_stk_name_dict

def stock_code_to_num(code):
    # 只保留数字部分
    return str(int(re.search(r'\d+', code).group()))

def check_stock_codes_in_data():
    # 获取映射字典
    bond_stk_dict = get_bond_stk_name_dict()
    stock_codes = set(str(int(re.search(r'\d+', code).group())) for code in bond_stk_dict.values())
    print(f"映射字典中共有 {len(stock_codes)} 个股票代码")
    
    # 检查几个不同时期的数据
    input_root = Path(r'D:\raw_data\stock_min_data_per_day')
    periods = {
        "2019年初": "20190102",
        "2019年中": "20190701",
        "2024年末": "20241210",
        "2025年初": "20250102"
    }
    
    for period_name, date_folder in periods.items():
        folder_path = input_root / date_folder
        if not folder_path.exists():
            print(f"{period_name} ({date_folder}) 文件夹不存在")
            continue
            
        data_codes = set()
        for file in folder_path.iterdir():
            if file.is_file():
                try:
                    df = pd.read_pickle(file)
                    code = str(df['code'].iloc[0])
                    data_codes.add(code)
                except Exception as e:
                    continue
        
        common_codes = stock_codes & data_codes
        print(f"\n{period_name} ({date_folder}):")
        print(f"- 数据中共有 {len(data_codes)} 个股票")
        print(f"- 其中 {len(common_codes)} 个股票在映射字典中")
        print(f"- 覆盖率: {len(common_codes)/len(data_codes)*100:.2f}%")
        
        if len(common_codes) < 5:
            print("数据示例:")
            print("- 数据中的股票代码:", list(data_codes)[:5])
            print("- 映射字典中的股票代码:", list(stock_codes)[:5])

if __name__ == "__main__":
    check_stock_codes_in_data() 