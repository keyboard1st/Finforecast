import pickle

import pandas as pd
from pymongo import MongoClient

from datetime import datetime


def wind_cbond_day(date):
    '''
    获取wind_cbond_day数据
    date: 日期，格式为'2025-04-10'
    '''
    client = MongoClient(
        'mongodb://chenhang:AZchenhang123@192.168.1.106:27017/?authMechanism=SCRAM-SHA-1&directConnection=true')
    database = client['zcs']
    collection = database['wind_cbond_day']
    cursor = collection.find({"date": date})
    wind_cbond_day_cur = pd.DataFrame(list(cursor))
    return wind_cbond_day_cur


def trade_day():
    client = MongoClient(
        'mongodb://chenhang:AZchenhang123@192.168.1.106:27017/?authMechanism=SCRAM-SHA-1&directConnection=true')
    database = client['zcs']
    collection = database['wind_trade_day']
    cursor = collection.find()
    wind_trade_day = pd.DataFrame(list(cursor)).date.values
    today = datetime.today()
    today = today.strftime("%Y-%m-%d")
    wind_trade_day = wind_trade_day[(wind_trade_day > '2019-01-01') & (wind_trade_day <= today)]
    wind_trade_day.sort()
    return wind_trade_day


def download_wind_cbond_basic():
    client = MongoClient(
        'mongodb://chenhang:AZchenhang123@192.168.1.106:27017/?authMechanism=SCRAM-SHA-1&directConnection=true')
    database = client['zcs']
    collection = database['wind_cbond_basic']
    cursor = collection.find()
    wind_cbond_basic = pd.DataFrame(list(cursor))
    return wind_cbond_basic


def get_bond_stk_name_dict():
    """
    获取转债与股票名称的对应。Key: bond, Value: stock
    """
    data = download_wind_cbond_basic()
    data = data[['code', 'underlying_code']].dropna(how='any')
    bond_stk_name_dict = {}
    for i in range(len(data)):
        bond_code = int(data.code.iloc[i][:6])
        stock_code = int(data.underlying_code.iloc[i][:6])
        bond_stk_name_dict[bond_code] = stock_code
    
    # 保存字典
    with open('bond_stk_name_dict.pkl', 'wb') as f:
        pickle.dump(bond_stk_name_dict, f)
    return bond_stk_name_dict

def get_bond_stk_name_df():
    """
    获取转债与股票名称的对应，返回DataFrame保持所有映射关系
    """
    data = download_wind_cbond_basic()
    data = data[['code', 'underlying_code']].dropna(how='any')
    
    # 直接创建DataFrame而不是字典
    mapping_data = []
    for i in range(len(data)):
        bond_code = int(data.code.iloc[i][:6])
        stock_code = int(data.underlying_code.iloc[i][:6])
        mapping_data.append({
            'bond_code': bond_code,
            'stock_code': stock_code
        })
    
    bond_stk_df = pd.DataFrame(mapping_data)
    
    # 按转债代码排序
    bond_stk_df = bond_stk_df.sort_values('bond_code').reset_index(drop=True)
    
    
    return bond_stk_df


def get_base_factor_enc():
    client = MongoClient(
        'mongodb://chenhang:AZchenhang123@192.168.1.106:27017/?authMechanism=SCRAM-SHA-1&directConnection=true')
    database = client['cb_factor_enc']
    collection = database['pricing1430']
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))