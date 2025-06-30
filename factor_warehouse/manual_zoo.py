import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
# 获取当前文件所在路径
current_dir = Path(__file__).resolve().parent
# 添加到 sys.path
sys.path.append(str(current_dir))
from get_data_from_pq import get_minute_ccb, get_minute_stock, get_daily_features, get_date_list


def manual_F1(date, assets):
# 取最后一分钟
    sc = get_minute_stock(date, 'close').iloc[-1]
    cc = get_minute_ccb(date, 'close').iloc[-1]
    swap = get_daily_features('swap_share_price').loc[date, assets]
    ratio = cc[assets] / sc[assets]
    return ratio/100 * swap - 1
def manual_F3(date, assets):
# 取最后一分钟
    cc = get_minute_ccb(date, 'close').iloc[-1]
    strbval = get_daily_features('strbvalue').loc[date, assets]
    return cc[assets] / strbval[assets] - 1
def manual_F5(date, assets):
    sc = get_minute_stock(date, 'close').iloc[-1]
    swap = get_daily_features('swap_share_price').loc[date, assets]
    strbval = get_daily_features('strbvalue').loc[date, assets]
    return 100 * sc[assets] / swap[assets] / strbval[assets] - 1
def manual_F7(date, assets):
    sc = get_minute_stock(date, 'close').iloc[-1]
    cc = get_minute_ccb(date, 'close').iloc[-1]
    swap = get_daily_features('swap_share_price').loc[date, assets]
    ratio = cc[assets] / sc[assets]
    return (ratio/100 * swap - 1) * 100 + cc[assets]

def manual_F9(date, assets):
    cc = get_minute_ccb(date, 'close').iloc[-1]
    f = cc - cc.shift(1)
    g = f.clip(lower=0)
    rsi = g.rolling(5).sum() / f.abs().rolling(5).sum()
    return rsi

def manual_F10(date, assets):
    cc = get_minute_ccb(date, 'close').iloc[-1]
    boll = cc.rolling(20, min_periods=10).mean()
    std = cc.rolling(20, min_periods=10).std()
    lower = boll - 2 * std
    upper = boll + 2 * std
    percent_b = (cc - lower) / (upper - lower)
    return percent_b

