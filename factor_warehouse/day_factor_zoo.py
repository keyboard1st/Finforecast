import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from factor_validator import FactorValidator

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
# 避免多线程下matplotlib默认使用tkinter导致的报错，只保存图片不弹窗
import matplotlib
matplotlib.use('Agg')

# 获取当前文件所在路径
current_dir = Path(__file__).resolve().parent
# 添加到 sys.path
sys.path.append(str(current_dir))
from get_data_from_pq import get_minute_ccb_all, get_minute_ccb, get_minute_stock, get_daily_features, get_date_list, get_daily_features_from_minute
from manual_zoo import *




def cal_F1():
    '''
    转股溢价率: LAST(可转债价格/正股价格)/100*转股价格 - 1， 取得是最后一分钟
    '''
    swap = get_daily_features('swap_share_price')

    ccb_close = get_daily_features_from_minute('ccb', 'close')
    stock_close = get_daily_features_from_minute('sk', 'close')
    df = ccb_close / stock_close
    
    swap_aligned = swap.reindex(index=df.index, columns=df.columns)
    
    f = df / 100 * swap_aligned - 1
    
    return f

def cal_F2():
    '''
    修正转股溢价率：转股价值X，真实转股溢价率Y，修正转股溢价率Z = Y - Y_hat = Y - (a+b/X)
    '''
    swap = get_daily_features('swap_share_price')

    ccb_close = get_daily_features_from_minute('ccb', 'close')
    stock_close = get_daily_features_from_minute('sk', 'close')
    ratio = ccb_close / stock_close
    
    swap_aligned = swap.reindex(index=ratio.index, columns=ratio.columns)
    
    Y = ratio / 100 * swap_aligned - 1
    X = 100 * stock_close / swap_aligned
    dates = X.index
    cols = X.columns

    Y_adj = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for t in dates:
        x = X.loc[t].values.astype(float)
        y = Y.loc[t].values.astype(float)
        mask = (x != 0) & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # 点太少，无法拟合
            Y_adj.loc[t] = np.nan
            continue

        w = (1.0 / x[mask]).reshape(-1, 1)
        M = np.hstack([w, np.ones_like(w)])
        y_obs = y[mask]

        try:
            # 尝试带截距拟合
            params, *_ = np.linalg.lstsq(M, y_obs, rcond=None)
            a, b = params
        except np.linalg.LinAlgError:
            # 奇异矩阵，退回到无截距模型
            a = (w.flatten() * y_obs).sum() / (w.flatten()**2).sum()
            b = 0.0

        # 计算残差
        fitted = a * (1.0 / x) + b
        Y_adj.loc[t] = y - fitted
    return Y_adj

def cal_F3():
    '''
    纯债溢价率: LAST(可转债价格)/纯债价值 - 1， 取得是最后一分钟
    '''
    dates = get_date_list()
    strbval = get_daily_features('strbvalue')
    cpu_count = os.cpu_count() or 16

    def cal_F3_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close').iloc[-1]
            return (date, ccb_close)
        except Exception as e:
            print(f"{date} 读取失败: {e}")
            return (date, pd.Series(np.nan, index=strbval.columns if hasattr(strbval, 'columns') else None))

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F3_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    
    f = pd.DataFrame({date: series for date, series in results}).T
    strbval_aligned = strbval.reindex(index=f.index, columns=f.columns)
    df = f / strbval_aligned - 1
    return df

def cal_F4():
    '''
    修正纯债溢价率：转股价值X，真实纯债溢价率Y，修正纯债溢价率Z = Y - Y_hat = Y - (aX + b)
    '''
    dates = get_date_list()
    swap = get_daily_features('swap_share_price')
    strbval = get_daily_features('strbvalue')
    cpu_count = os.cpu_count() or 16

    def cal_F4_one_day(date):
        try:
            stock_close = get_minute_stock(date, 'close')
            ccb_close = get_minute_ccb(date, 'close')
            stock_last = stock_close.iloc[-1]
            ccb_close = ccb_close.iloc[-1]
            return (date, ccb_close, stock_last)
        except Exception as e:
            print(f"{date} 读取失败: {e}")
            return (date, pd.Series(np.nan, index=swap.columns if hasattr(swap, 'columns') else None), pd.Series(np.nan, index=swap.columns if hasattr(swap, 'columns') else None))

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F4_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    
    stock_last = pd.DataFrame({date: series for date, _, series in results}).T
    ccb_close = pd.DataFrame({date: series for date, series, _ in results}).T
    
    strbval_aligned = strbval.reindex(index=stock_last.index, columns=stock_last.columns)
    swap_aligned = swap.reindex(index=stock_last.index, columns=stock_last.columns)
    
    Y = ccb_close / strbval_aligned - 1
    X = 100 * stock_last / swap_aligned
    all_cols = X.columns.union(Y.columns)
    X = X.reindex(columns=all_cols)
    Y = Y.reindex(columns=all_cols)
    dates = X.index
    cols = X.columns

    Y_adj = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for t in dates:
        x = X.loc[t].values.astype(float)
        y = Y.loc[t].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # 点太少，无法拟合
            Y_adj.loc[t] = np.nan
            continue
        x_valid = x[mask].reshape(-1, 1)
        y_valid = y[mask]
        M = np.hstack([x_valid, np.ones_like(x_valid)])
        try:
            params, *_ = np.linalg.lstsq(M, y_valid, rcond=None)
            a, b = params
        except np.linalg.LinAlgError:
            a = 0.0
            b = y_valid.mean()
        fitted = a * x + b
        Y_adj.loc[t] = y - fitted
    return Y_adj
    
def cal_F5():
    '''
    转股纯债溢价率: 转股价值/纯债价值 - 1
    '''
    dates = get_date_list()
    swap = get_daily_features('swap_share_price')
    strbval = get_daily_features('strbvalue')
    cpu_count = os.cpu_count() or 16

    def cal_F5_one_day(date):
        try:
            stock_close = get_minute_stock(date, 'close')
            stock_last = stock_close.iloc[-1]   # 取last和mean没区别
            return (date, stock_last)
        except Exception as e:
            print(f"{date} 读取失败: {e}")
            return (date, pd.Series(np.nan, index=swap.columns if hasattr(swap, 'columns') else None))

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F5_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    
    stock_last = pd.DataFrame({date: series for date, series in results}).T
    strbval_aligned = strbval.reindex(index=stock_last.index, columns=stock_last.columns)
    swap_aligned = swap.reindex(index=stock_last.index, columns=stock_last.columns)
    
    X = 100 * stock_last / swap_aligned
    f = X / strbval_aligned - 1
    return f

def cal_F6():
    '''
    双低因子: 转股溢价率 * 100 + 可转债价格
    '''
    dates = get_date_list()
    swap = get_daily_features('swap_share_price')
    cpu_count = os.cpu_count() or 16

    def cal_F6_one_day(date):
        try:
            stock_close = get_minute_stock(date, 'close')
            ccb_close = get_minute_ccb(date, 'close')
            stock_last = stock_close.iloc[-1]
            ccb_close = ccb_close.iloc[-1]
            ratio = ccb_close / stock_last
            return (date, ratio, ccb_close)
        except Exception as e:
            print(f"{date} 读取失败: {e}")
            return (date, pd.Series(np.nan, index=swap.columns if hasattr(swap, 'columns') else None), pd.Series(np.nan, index=swap.columns if hasattr(swap, 'columns') else None))

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F6_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    
    ratio = pd.DataFrame({date: series for date, series, _ in results}).T
    ccb_close = pd.DataFrame({date: series for date, _, series in results}).T
    
    swap_aligned = swap.reindex(index=ratio.index, columns=ratio.columns)
    
    f = (ratio * swap_aligned / 100 - 1) * 100 + ccb_close
    
    return f

def cal_F7():
    '''
    修正双低因子： std(修正转股溢价率) + std(可转债价格)
    '''
    dates = get_date_list()
    swap = get_daily_features('swap_share_price')
    cpu_count = os.cpu_count() or 16

    def cal_F7_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_close = ccb_close.iloc[-1]
            return (date, ccb_close)
        except Exception as e:
            print(f"{date} 读取失败: {e}")
            return (date, pd.Series(np.nan, index=swap.columns if hasattr(swap, 'columns') else None))

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F7_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    
    ccb_close = pd.DataFrame({date: series for date, series in results}).T
    ccb_std = (ccb_close.sub(ccb_close.mean(axis=1), axis=0)).div(ccb_close.std(axis=1), axis=0)
    f2 = cal_F2()
    f2_std = (f2.sub(f2.mean(axis=1), axis=0)).div(f2.std(axis=1), axis=0)
    f2_std.replace([np.inf, -np.inf], np.nan, inplace=True)
    ccb_std.replace([np.inf, -np.inf], np.nan, inplace=True)
    f7 = f2_std + ccb_std
    return f7

def cal_F8():
    '''
    动量5日： 5日收益率
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F8_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_close = ccb_close.iloc[-1]
            return (date, ccb_close)
        except Exception as e:
            print(f"{date} 读取失败: {e}")
            return (date, pd.Series(np.nan, index=swap.columns if hasattr(swap, 'columns') else None))

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F8_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    
    ccb_close = pd.DataFrame({date: series for date, series in results}).T

    f = ccb_close.pct_change(5)
    return f
    
def cal_F9():
    '''
    RSI: 5日stock涨幅之和 / 5日stock涨跌幅绝对值之和
    '''
    sk_close = get_daily_features_from_minute('sk', 'close')
    delta = sk_close - sk_close.shift(1)
    gain = delta.clip(lower=0)  # 涨的部分，跌的设为0
    rsi = gain.rolling(5).sum() / delta.abs().rolling(5).sum()
    return rsi


def cal_F10():
    '''
    Percent B: (stock收盘价 - 20日布林线下轨) / 20日布林带宽度
    '''
    sk_close = get_daily_features_from_minute('sk', 'close')
    boll = sk_close.rolling(20, min_periods=10).mean()
    std = sk_close.rolling(20, min_periods=10).std()
    upper = boll + 2 * std
    lower = boll - 2 * std
    percent_b = (sk_close - lower) / (upper - lower)
    return percent_b
    
def cal_F11():
    '''
    Price to High: stock收盘价 / 过去20日的stock最高价
    '''
    sk_close = get_daily_features_from_minute('sk', 'close')
    sk_high = get_daily_features_from_minute('sk', 'high')
    f = sk_close / sk_high.rolling(20, min_periods=10).max()
    return f

def cal_F12():
    '''
    Amihud: MEAN(abs(stock收盘价 - stock前收盘价) / stock成交量)
    '''
    sk_close = get_daily_features_from_minute('sk', 'close')
    sk_volume = get_daily_features_from_minute('sk', 'volume')
    amihud = (sk_close - sk_close.shift(1)).abs() / sk_volume
    amihud.replace([np.inf, -np.inf], np.nan, inplace=True)
    return amihud

def cal_F13():
    '''
    日度资金流: ((stock_high + stock_low + stock_close) / 3) * stock_volume
    5日资金流比率 = 5日正资金流之和 / 5日负资金流之和
    5日MFI = 100 - 100 / (1 + 5日资金流比率)
    '''
    sk_high = get_daily_features_from_minute('sk', 'high')
    sk_low = get_daily_features_from_minute('sk', 'low')
    sk_close = get_daily_features_from_minute('sk', 'close')
    sk_volume = get_daily_features_from_minute('sk', 'volume')
    
    typ = (sk_high + sk_low + sk_close) / 3
    typ_shift = typ.shift(1)
    buy_vol = (typ * sk_volume).where(typ > typ_shift, 0)
    sell_vol = (typ * sk_volume).where(typ < typ_shift, 0)
    ratio = buy_vol.rolling(5).sum() / sell_vol.rolling(5).sum()
    ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
    mfi = 100 - 100 / (1 + ratio)
    return mfi

def cal_F14():
    '''
    日内(转债与正股涨跌幅之差)的累计求和
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F14_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            sk_close = get_minute_stock(date, 'close')
            ccb_ret = ccb_close.pct_change(1, fill_method=None)
            sk_ret = sk_close.pct_change(1, fill_method=None)
            f = (ccb_ret - sk_ret).sum()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F14_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    
    f = pd.DataFrame({date: series for date, series in results}).T
    return f

def cal_F15():
    '''
    日间 转债与正股涨跌幅之差
    '''
    sk_close = get_daily_features_from_minute('sk', 'close')
    ccb_close = get_daily_features_from_minute('ccb', 'close')
    f = sk_close.pct_change(1, fill_method=None) - ccb_close.pct_change(1, fill_method=None)
    return f



def cal_F16():
    '''
    日内转债与正股涨跌幅的相关性
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F16_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            sk_close = get_minute_stock(date, 'close')
            ccb_ret = ccb_close.pct_change(1, fill_method=None)
            sk_ret = sk_close.pct_change(1, fill_method=None)
            f = ccb_ret.corrwith(sk_ret)
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F16_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    
    f = pd.DataFrame({date: series for date, series in results}).T
    return f

def cal_F17():
    '''
    日内温和收益：首先计算日内分钟数据对数收益率的中位数和 MAD，在中位数 1.96 倍 MAD 以内的分钟线定义为温和收益，将每日所有的温和对数收益率相加，得到温和收益。
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F17_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_ret = np.log(ccb_close / ccb_close.shift(1))
            median = ccb_ret.median()
            mad = (ccb_ret - median).abs().median()
            f = ccb_ret.where(ccb_ret.abs() < median + 1.96 * mad, 0).sum()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F17_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    return f

def cal_F18():
    '''
    分钟线方差均值：Mean(30分钟收益率方差)
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F18_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_ret = np.log(ccb_close / ccb_close.shift(1))
            grouped_std = (ccb_ret.groupby(np.arange(len(ccb_ret)) // 30).std())
            f = grouped_std.mean()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F18_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    return f

def cal_F19():
    '''
    分钟线偏度均值：Mean(30分钟收益率偏度)
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F19_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_ret = np.log(ccb_close / ccb_close.shift(1))
            grouped_skew = (ccb_ret.groupby(np.arange(len(ccb_ret)) // 30).skew())
            f = grouped_skew.mean()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F19_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    return f

def cal_F20():
    '''
    分钟线均值方差：STD(30分钟收益率均值)
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F20_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_ret = np.log(ccb_close / ccb_close.shift(1))
            grouped_std = (ccb_ret.groupby(np.arange(len(ccb_ret)) // 30).mean())
            f = grouped_std.std()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F20_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    return f


def cal_F21():
    '''
    分钟线偏度方差：STD(30分钟收益率偏度)
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F21_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_ret = np.log(ccb_close / ccb_close.shift(1))
            grouped_skew = (ccb_ret.groupby(np.arange(len(ccb_ret)) // 30).skew())
            f = grouped_skew.std()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F21_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    return f

def cal_F22():
    '''
    分钟线方差方差：STD(30分钟收益率方差)
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F22_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_ret = np.log(ccb_close / ccb_close.shift(1))
            grouped_std = (ccb_ret.groupby(np.arange(len(ccb_ret)) // 30).std())
            f = grouped_std.std()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F22_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    return f

def cal_F23():
    '''
    收盘价与成交量相关系数
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F23_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ccb_volume = get_minute_ccb(date, 'volume')
            f = ccb_close.corrwith(ccb_volume)
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F23_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    return f


def cal_F24():
    '''
    早盘成交量占比
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F24_one_day(date):
        try:
            ccb_volume = get_minute_ccb(date, 'volume')
            mask = ccb_volume.index.values % 1_000_000 <= 100000
            f = ccb_volume[mask].sum() / ccb_volume.sum()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F24_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    return f

def cal_F25():
    '''
    尾盘成交量占比
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F25_one_day(date):
        try:
            ccb_volume = get_minute_ccb(date, 'volume')
            mask = ccb_volume.index.values % 1_000_000 > 140000
            f = ccb_volume[mask].sum() / ccb_volume.sum()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F25_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    return f

def cal_F26():
    '''
    早盘收益率占比
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F26_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ret = np.log(ccb_close / ccb_close.shift(1))
            mask = ret.index.values % 1_000_000 <= 100000
            f = ret[mask].sum() / ret.sum()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F26_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f

def cal_F27():
    '''
    尾盘收益率占比
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F27_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ret = np.log(ccb_close / ccb_close.shift(1))
            mask = ret.index.values % 1_000_000 > 140000
            f = ret[mask].sum() / ret.sum()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F27_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f

def cal_F28():
    '''
    日内收益率>0位置上的交易量占比
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F28_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            vol = get_minute_ccb(date, 'volume')
            ret = np.log(ccb_close / ccb_close.shift(1))
            mask = ret > 0
            f = vol[mask].sum() / vol.sum()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")
            
    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F28_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f

def cal_F29():
    '''
    已实现波动率：日内收益率平方之和
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F29_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ret = np.log(ccb_close / ccb_close.shift(1))
            f = ret.pow(2).sum()
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")
        
    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F29_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))

    f = pd.DataFrame({date: series for date, series in results}).T
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f

def cal_F30():
    '''
    （正收益率计算已实现波动率-负收益计算已实现波动率）/已实现波动率
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F30_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ret = np.log(ccb_close / ccb_close.shift(1))
            f = ret.pow(2)
            pos_ret = f[ret > 0]
            neg_ret = f[ret < 0]
            df = (pos_ret.sum() - neg_ret.sum()) / f.sum()
            return (date, df)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")
    
    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F30_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    f = pd.DataFrame({date: series for date, series in results}).T
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f

def cal_F31():
    '''
    将当天分钟频率数据按照收益率正负划分，计算正收益 std 与负收益 std 差值
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F31_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ret = np.log(ccb_close / ccb_close.shift(1))
            pos_ret = ret[ret > 0]
            neg_ret = ret[ret < 0]
            df = (pos_ret.std() - neg_ret.std())
            return (date, df)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")
        
    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F31_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    f = pd.DataFrame({date: series for date, series in results}).T
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f


def cal_F32():
    '''
    计算日内1分钟收益率排序前后20%部分累计收益率 的 差 作为因子值
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F32_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ret = np.log(ccb_close / ccb_close.shift(1))
            top_20 = ret.quantile(0.8, axis=0)
            bottom_20 = ret.quantile(0.2, axis=0)
            f = ret[ret >= top_20].sum(axis=0) - ret[ret <= bottom_20].sum(axis=0)
            return (date, f)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")
        
    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F32_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    f = pd.DataFrame({date: series for date, series in results}).T
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f


def cal_F33():
    '''
    计算除法和Log差值算法的两个1分钟收益率
    每分钟计算2*(除法收益率-Log收益率)-Log收益率^2，并除以除法收益率绝对值全日均值
    统计日内每分钟均值作为当天的因子值
    '''
    dates = get_date_list()
    cpu_count = os.cpu_count() or 16
    def cal_F33_one_day(date):
        try:
            ccb_close = get_minute_ccb(date, 'close')
            ln_ret = np.log(ccb_close / ccb_close.shift(1))
            div_ret = ccb_close / ccb_close.shift(1) - 1
            f = (2*(div_ret - ln_ret) - ln_ret.pow(2)) / div_ret.abs().mean()
            df = f.mean()
            return (date, df)
        except Exception as e:
            raise Exception(f"{date} 读取失败: {e}")
    
    results = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_date = {executor.submit(cal_F33_one_day, date): date for date in dates}
        for future in as_completed(future_to_date):
            results.append(future.result())
    results.sort(key=lambda x: dates.index(x[0]))
    f = pd.DataFrame({date: series for date, series in results}).T
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f




if __name__ == '__main__':
    label = pd.read_parquet(r'D:\chenxing\Finforecast\factor_warehouse\label\label_vwap_log_cliped')
    f33 = cal_F33()
    
    # validator = FactorValidator(
    #     factor=f11,
    #     manual_check=manual_F11,
    #     sample_n=10,
    #     plot_path=r"D:\chenxing\Finforecast\factor_warehouse\plots\F11",
    #     label=label
    # )
    # f11_df = validator.validate()

    validator = FactorValidator(
        factor=f33,
        plot_path=r"D:\chenxing\Finforecast\factor_warehouse\plots\F33",
        label=label
    )
    f33_df = validator.validate()







