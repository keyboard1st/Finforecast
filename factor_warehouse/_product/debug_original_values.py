import pandas as pd
import numpy as np

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

def debug_original_values():
    """详细检查原始数据值"""
    print("详细检查原始数据值...")
    
    test_dates = ['2019-01-02', '2019-01-03', '2019-01-04']
    
    for date in test_dates:
        print(f"\n{'='*60}")
        print(f"日期: {date}")
        
        df = wind_cbond_day(date)
        
        if df.empty:
            print("❌ 没有数据")
            continue
        
        # 检查前5个可转债的具体值
        print("前5个可转债的swap_share_price值:")
        for i in range(min(5, len(df))):
            code = df.iloc[i]['code']
            price = df.iloc[i]['swap_share_price']
            print(f"  {code}: {price}")
        
        # 检查是否有重复值
        prices = df['swap_share_price'].values
        unique_prices = np.unique(prices)
        print(f"\n价格统计:")
        print(f"  总记录数: {len(prices)}")
        print(f"  唯一价格数: {len(unique_prices)}")
        print(f"  重复率: {(len(prices) - len(unique_prices)) / len(prices):.2%}")
        
        # 检查前10个唯一价格
        print(f"  前10个唯一价格: {sorted(unique_prices)[:10]}")
        
        # 检查是否有异常值
        if len(unique_prices) < 10:
            print(f"  ⚠ 警告: 唯一价格过少!")
            print(f"  所有唯一价格: {sorted(unique_prices)}")

if __name__ == '__main__':
    debug_original_values() 