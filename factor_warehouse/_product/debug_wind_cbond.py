import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def wind_cbond_day(date):
    '''
    获取wind_cbond_day数据
    date: 日期，格式为'2025-04-10'
    '''
    try:
        from pymongo import MongoClient
        client = MongoClient(
            'mongodb://chenhang:AZchenhang123@192.168.1.106:27017/?authMechanism=SCRAM-SHA-1&directConnection=true')
        database = client['zcs']
        collection = database['wind_cbond_day']
        cursor = collection.find({"date": date})
        wind_cbond_day_cur = pd.DataFrame(list(cursor))
        client.close()
        return wind_cbond_day_cur
    except Exception as e:
        print(f"Error fetching data for {date}: {e}")
        return pd.DataFrame()

def debug_original_data():
    """调试原始数据"""
    print("开始调试wind_cbond_day原始数据...")
    
    # 测试几个不同的日期
    test_dates = ['2019-01-02', '2019-01-03', '2019-01-04', '2020-01-02', '2021-01-02']
    
    for date in test_dates:
        print(f"\n{'='*50}")
        print(f"检查日期: {date}")
        
        df = wind_cbond_day(date)
        
        if df.empty:
            print(f"❌ {date} 没有数据")
            continue
            
        print(f"✅ 数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 检查code字段
        if 'code' in df.columns:
            print(f"code字段示例: {df['code'].head().tolist()}")
            print(f"code字段类型: {df['code'].dtype}")
            print(f"code字段唯一值数量: {df['code'].nunique()}")
        
        # 检查swap_share_price字段
        if 'swap_share_price' in df.columns:
            price_series = df['swap_share_price']
            print(f"swap_share_price统计:")
            print(f"  非空值数量: {price_series.notna().sum()}")
            print(f"  唯一值数量: {price_series.nunique()}")
            print(f"  最小值: {price_series.min()}")
            print(f"  最大值: {price_series.max()}")
            print(f"  前10个值: {price_series.head(10).tolist()}")
            
            # 检查是否有异常值
            if price_series.nunique() < 5:
                print(f"  ⚠ 警告: swap_share_price唯一值过少!")
                print(f"  唯一值: {sorted(price_series.dropna().unique())}")
        
        # 显示前几行数据
        print(f"前5行数据:")
        print(df.head())
        
        # 检查数据类型
        print(f"数据类型:")
        print(df.dtypes)

def check_data_processing():
    """检查数据处理过程"""
    print(f"\n{'='*50}")
    print("检查数据处理过程...")
    
    # 模拟数据处理过程
    date = '2019-01-02'
    df = wind_cbond_day(date)
    
    if df.empty:
        print("❌ 无法获取测试数据")
        return
    
    print(f"原始数据形状: {df.shape}")
    
    # 模拟process_wind_cbond_features.py中的处理过程
    if 'code' in df.columns and 'swap_share_price' in df.columns:
        # 处理code字段
        df_temp = df[['code', 'swap_share_price']].copy()
        print(f"处理前code示例: {df_temp['code'].head().tolist()}")
        
        # 去掉后缀
        df_temp['code'] = df_temp['code'].astype(str).str.replace('.SH', '').str.replace('.SZ', '').str.replace('.BJ', '')
        print(f"处理后code示例: {df_temp['code'].head().tolist()}")
        
        # 转换为整数
        df_temp['code'] = df_temp['code'].astype(int)
        print(f"转换为整数后: {df_temp['code'].head().tolist()}")
        
        # 设置为索引
        result = df_temp.set_index('code')['swap_share_price']
        print(f"最终结果:")
        print(f"  形状: {result.shape}")
        print(f"  非空值: {result.notna().sum()}")
        print(f"  唯一值: {result.nunique()}")
        print(f"  前10个值: {result.head(10).tolist()}")

if __name__ == '__main__':
    debug_original_data()
    check_data_processing() 