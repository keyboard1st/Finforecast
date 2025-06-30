import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 修复wind_cbond_day函数的bug
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

def generate_date_range(start_date='2019-01-02', end_date='2025-04-11'):
    """生成日期范围"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates

def process_wind_cbond_features():
    """处理wind_cbond特征数据"""
    print("开始处理wind_cbond特征数据...")
    
    # 生成日期范围
    dates = generate_date_range('2019-01-02', '2025-04-11')
    print(f"日期范围: {dates[0]} 到 {dates[-1]}, 共 {len(dates)} 天")
    
    # 存储所有特征数据
    feature_data = {}
    valid_dates = []
    
    # 遍历每个日期获取数据
    for date in tqdm(dates, desc="获取数据"):
        df = wind_cbond_day(date)
        
        if df.empty:
            print(f"警告: {date} 没有数据")
            continue
            
        # 检查必要的列是否存在
        if 'code' not in df.columns:
            print(f"警告: {date} 缺少code列")
            continue
            
        # 转换日期格式为8位整数
        date_int = int(date.replace('-', ''))
        valid_dates.append(date_int)
        
        # 处理每个特征列
        for col in df.columns:
            if col in ['_id', 'date', 'code']:  # 跳过元数据列
                continue
                
            if col not in feature_data:
                feature_data[col] = {}
            
            # 将code转换为6位整数，并设置为索引
            df_temp = df[['code', col]].copy()
            # 处理code字段，去掉后缀（如.SH）
            df_temp['code'] = df_temp['code'].astype(str).str.replace('.SH', '').str.replace('.SZ', '').str.replace('.BJ', '')
            df_temp['code'] = df_temp['code'].astype(int)
            df_temp = df_temp.set_index('code')[col]
            
            # 存储到特征数据中
            feature_data[col][date_int] = df_temp
    
    print(f"成功获取 {len(valid_dates)} 个有效日期的数据")
    print(f"发现 {len(feature_data)} 个特征")
    
    # 创建输出目录
    output_dir = r'D:\chenxing\Finforecast\factor_warehouse\daily_parquet'
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个特征创建DataFrame并保存
    for feature_name in tqdm(feature_data.keys(), desc="保存特征文件"):
        try:
            # 创建特征DataFrame
            feature_df = pd.DataFrame(feature_data[feature_name]).T
            
            # 设置索引和列的数据类型
            feature_df.index = feature_df.index.astype(int)  # 日期为8位整数
            feature_df.columns = feature_df.columns.astype(int)  # code为6位整数
            
            # 按日期排序
            feature_df = feature_df.sort_index()
            
            # 保存为parquet文件
            output_path = os.path.join(output_dir, f'{feature_name}.parquet')
            feature_df.to_parquet(output_path)
            
            print(f"✓ 保存特征 {feature_name}: 形状 {feature_df.shape}")
            
        except Exception as e:
            print(f"✗ 保存特征 {feature_name} 失败: {e}")
    
    print(f"\n所有特征文件已保存到: {output_dir}")
    
    # 验证输出文件
    print("\n验证输出文件:")
    for feature_name in list(feature_data.keys())[:5]:  # 只验证前5个特征
        try:
            output_path = os.path.join(output_dir, f'{feature_name}.parquet')
            if os.path.exists(output_path):
                df_check = pd.read_parquet(output_path)
                print(f"✓ {feature_name}: 形状 {df_check.shape}, 索引类型 {type(df_check.index[0])}, 列类型 {type(df_check.columns[0])}")
            else:
                print(f"✗ {feature_name}: 文件不存在")
        except Exception as e:
            print(f"✗ {feature_name}: 验证失败 - {e}")

if __name__ == '__main__':
    process_wind_cbond_features() 