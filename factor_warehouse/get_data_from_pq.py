import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

'''
not_consider_bonds = [121002, 121003, 132004, 132006, 132007, 132008, 132009, 132011, 132014, 132015, 132016, 132017, 132018, 132021, 132022]
'''

def filter_time_before_1430(df):
    time_part = df.index % 1_000_000  # 取末尾6位，即HHMMSS

    # 筛选时间早于 14:30:00（143000）
    filtered_df = df[time_part <= 143000]
    return filtered_df

def filter_cols(df):
    not_consider_bonds = [121002, 121003, 132004, 132006, 132007, 132008, 132009, 132011, 132014, 132015, 132016, 132017, 132018, 132021, 132022]
    df = df.drop(columns=not_consider_bonds, errors='ignore')
    return df

def get_minute_ccb(date, col:str):
    '''
    date: int, 20250411
    col: str, 'close'
    '''
    folder_path = os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_one_minute_parquet',str(date))
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f'{folder_path} not found')
    df = pd.read_parquet(os.path.join(folder_path,f'{col}.parquet'))
    df = filter_time_before_1430(df)
    df = filter_cols(df)
    return df

def get_minute_ccb_all(date, col:str):
    '''
    get all data of one day
    date: int, 20250411
    col: str, 'close'
    '''
    folder_path = os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_one_minute_parquet',str(date))
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f'{folder_path} not found')
    df = pd.read_parquet(os.path.join(folder_path,f'{col}.parquet'))
    df = filter_cols(df)
    return df

def get_minute_stock(date, col:str):
    '''
    date: int, 20250411
    col: str, 'close'
    '''
    folder_path = os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_one_minute_parquet',str(date))
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f'{folder_path} not found')
    df = pd.read_parquet(os.path.join(folder_path,f'stock_{col}.parquet'))
    df = filter_time_before_1430(df)
    df = filter_cols(df)
    return df


def get_date_list():
    folder = r'D:\chenxing\Finforecast\factor_warehouse\data_one_minute_parquet'
    date_list = [int(date) for date in sorted(os.listdir(folder))]
    return date_list

def get_daily_features(col):
    path = os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_daily_parquet',f'{col}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found')
    df = pd.read_parquet(path)
    return df

def get_daily_features_from_minute(type_f, col):
    '''
    col: str, 'close'
    type: str, 'ccb' or 'sk'
    '''
    try:
        path = os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_useful_daily_parquet',type_f + '_' + col + '.parquet')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found')
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        raise Exception(f"{type_f} {col} 读取失败: {e}")



def save_daily_features():
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_daily_features(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_close = ccb_close.iloc[-1]
            ccb_volume = get_minute_ccb(date, 'volume')
            ccb_volume = ccb_volume.sum()
            ccb_high = get_minute_ccb(date, 'high')
            ccb_high = ccb_high.max()
            ccb_low = get_minute_ccb(date, 'low')
            ccb_low = ccb_low.min()
            sk_close = get_minute_stock(date, 'close')
            sk_close = sk_close.iloc[-1]
            sk_volume = get_minute_stock(date, 'volume')
            sk_volume = sk_volume.sum()
            sk_high = get_minute_stock(date, 'high')
            sk_high = sk_high.max()
            sk_low = get_minute_stock(date, 'low')
            sk_low = sk_low.min()

            return (date, ccb_close, ccb_volume, ccb_high, ccb_low, sk_close, sk_volume, sk_high, sk_low)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_daily_features, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    
    ccb_close = pd.DataFrame({date: series for date, series, _, _, _, _, _, _, _ in results}).T
    ccb_volume = pd.DataFrame({date: series for date, _, series, _, _, _, _, _, _ in results}).T
    ccb_high = pd.DataFrame({date: series for date, _, _, series, _, _, _, _, _ in results}).T
    ccb_low = pd.DataFrame({date: series for date, _, _, _, series, _, _, _, _ in results}).T
    sk_close = pd.DataFrame({date: series for date, _, _, _, _, series, _, _, _ in results}).T
    sk_volume = pd.DataFrame({date: series for date, _, _, _, _, _, series, _, _ in results}).T
    sk_high = pd.DataFrame({date: series for date, _, _, _, _, _, _, series, _ in results}).T
    sk_low = pd.DataFrame({date: series for date, _, _, _, _, _, _, _, series in results}).T

    ccb_close.to_parquet(os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_useful_daily_parquet',f'ccb_close.parquet'))
    ccb_volume.to_parquet(os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_useful_daily_parquet',f'ccb_volume.parquet'))
    ccb_high.to_parquet(os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_useful_daily_parquet',f'ccb_high.parquet'))
    ccb_low.to_parquet(os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_useful_daily_parquet',f'ccb_low.parquet'))
    sk_close.to_parquet(os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_useful_daily_parquet',f'sk_close.parquet'))
    sk_volume.to_parquet(os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_useful_daily_parquet',f'sk_volume.parquet'))
    sk_high.to_parquet(os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_useful_daily_parquet',f'sk_high.parquet'))
    sk_low.to_parquet(os.path.join(r'D:\chenxing\Finforecast\factor_warehouse\data_useful_daily_parquet',f'sk_low.parquet'))
    print('daily_features saved')
    return 

if __name__ == '__main__':
    save_daily_features()



