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

def test_data_processing():
    """测试数据处理过程"""
    print("测试数据处理过程...")
    
    # 测试连续几天的数据
    test_dates = ['2019-01-02', '2019-01-03', '2019-01-04']
    feature_data = {}
    
    for date in test_dates:
        print(f"\n处理日期: {date}")
        df = wind_cbond_day(date)
        
        if df.empty:
            continue
            
        date_int = int(date.replace('-', ''))
        
        # 处理swap_share_price
        col = 'swap_share_price'
        if col not in feature_data:
            feature_data[col] = {}
        
        # 将code转换为6位整数，并设置为索引
        df_temp = df[['code', col]].copy()
        df_temp['code'] = df_temp['code'].astype(str).str.replace('.SH', '').str.replace('.SZ', '').str.replace('.BJ', '')
        df_temp['code'] = df_temp['code'].astype(int)
        df_temp = df_temp.set_index('code')[col]
        
        print(f"  原始数据形状: {df.shape}")
        print(f"  处理后数据形状: {df_temp.shape}")
        print(f"  唯一值数量: {df_temp.nunique()}")
        print(f"  前5个值: {df_temp.head().tolist()}")
        
        # 存储到特征数据中
        feature_data[col][date_int] = df_temp
    
    # 创建DataFrame
    print(f"\n创建DataFrame...")
    feature_df = pd.DataFrame(feature_data['swap_share_price']).T
    print(f"DataFrame形状: {feature_df.shape}")
    print(f"DataFrame列数: {len(feature_df.columns)}")
    print(f"DataFrame行数: {len(feature_df)}")
    
    # 检查每列的变化情况
    print(f"\n检查每列的变化情况:")
    for col in feature_df.columns[:5]:
        series = feature_df[col].dropna()
        if len(series) > 0:
            unique_vals = series.nunique()
            change_rate = unique_vals / len(series)
            print(f"列 {col}: 非空值 {len(series)}, 唯一值 {unique_vals}, 变化率 {change_rate:.2%}")
            if change_rate < 0.5:
                print(f"  值: {sorted(series.unique())}")
    
    # 显示DataFrame内容
    print(f"\nDataFrame内容:")
    print(feature_df.head())

if __name__ == '__main__':
    test_data_processing() 