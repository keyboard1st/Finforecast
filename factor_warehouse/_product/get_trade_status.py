import pandas as pd
from pymongo import MongoClient
import numpy as np

def wind_cbond_day(date):
    '''
    获取wind_cbond_day数据
    date: 日期，格式为'2025-04-10'
    '''
    client = MongoClient(
        'mongodb://chenhang:AZchenhang123@192.168.1.106:27017/?authMechanism=SCRAM-SHA-1&directConnection=true')
    database = client['zcs']
    collection = database['wind_cbond_day']
    cursor = collection.find({"date": date}, {"trade_status": 1, "code": 1, "_id": 0})
    wind_cbond_day_cur = pd.DataFrame(list(cursor))
    return wind_cbond_day_cur

def get_all_trade_status():
    '''
    遍历所有日期，获取trade_status数据并生成宽表
    '''
    # 读取label文件获取日期列表
    label_df = pd.read_parquet('factor_warehouse/label_raw/label_vwap_log_cliped_new.parquet')
    dates = label_df.index.tolist()
    
    # 存储所有日期的数据
    all_data = []
    
    for date in dates:
        print(f"处理日期: {date}")
        # 将日期转换为MongoDB查询格式 (YYYY-MM-DD)
        date_str = str(date)
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        try:
            # 获取当日数据
            daily_data = wind_cbond_day(formatted_date)
            
            if not daily_data.empty:
                # 将code设为索引，trade_status为值
                daily_series = daily_data.set_index('code')['trade_status']
                daily_series.name = date
                all_data.append(daily_series)
            else:
                print(f"警告: 日期 {formatted_date} 没有数据")
                
        except Exception as e:
            print(f"错误: 处理日期 {formatted_date} 时出错: {e}")
    
    # 合并所有数据
    if all_data:
        result_df = pd.concat(all_data, axis=1).T
        print(f"生成宽表完成，形状: {result_df.shape}")
        return result_df
    else:
        print("没有获取到任何数据")
        return pd.DataFrame()

# 执行并保存结果
if __name__ == "__main__":
    trade_status_df = get_all_trade_status()
    
    if not trade_status_df.empty:
        # 保存为parquet文件
        output_path = 'factor_warehouse/label_raw/trade_status_wide.parquet'
        trade_status_df.to_parquet(output_path)
        print(f"结果已保存到: {output_path}")
        
        # 显示基本信息
        print(f"\n数据形状: {trade_status_df.shape}")
        print(f"日期范围: {trade_status_df.index.min()} - {trade_status_df.index.max()}")
        print(f"股票代码数量: {len(trade_status_df.columns)}")
        print("\n前5行前5列数据:")
        print(trade_status_df.iloc[:5, :5])
    else:
        print("未能生成数据") 