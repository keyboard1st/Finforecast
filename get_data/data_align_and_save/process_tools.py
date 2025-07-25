import os
import sys

import numpy as np
import pandas as pd



def align_fix_with_market(factor_value:pd.DataFrame, marketcap:pd.DataFrame):
    '''
        因子值与marketcap对齐并填均(裁剪)值然后mask.
        Params:
            factor_value: 待对齐的因子值
            marketcap: 待对齐的marketcap
        Returns
        ---
        factor_value
            对齐后的因子值
    '''
    # marketcap = market_3d['marketcap']
    # common_index = factor_value.index.union(marketcap.index)
    common_cols = factor_value.columns.union(marketcap.columns)
    factor_value = factor_value.reindex(index=marketcap.index, columns=common_cols)
    marketcap = marketcap.reindex(index=marketcap.index, columns=common_cols)
    factor_value = factor_value.apply(adjust_row, axis=1)
    factor_value = factor_value.where(marketcap>0)
    
    return factor_value

def align_with_label(factor_value:pd.DataFrame, label:pd.DataFrame):
    '''
        与label对齐索引和列名(无裁剪、填均值等操作). 最好用于Dataloader.
    '''
    return factor_value.reindex_like(label).where(~(label.isna()))

def fillna_aligned(df_aligned:pd.DataFrame, label:pd.DataFrame,
                   low:float=0.05,
                   high:float=0.95
                   ):
    '''
        这个函数最好在Dataloader里使用.
        Params:
            df_aligned: 与marketcap对齐好并fillna、mask的因子值。
            label: y_label, 最终对齐的标签。
            low: 裁剪填充的低分位数.
            high: 裁剪填充的高分位数.
        Returns
        ---
        DataFrame
            与label对齐好并fillna、mask的因子值。
    '''
    clipped_df = df_aligned.apply(lambda row: 
    row.clip(
        lower=row.quantile(low), 
        upper=row.quantile(high)
    ), 
    axis=1
)
    mask = label.notna() & clipped_df.isna()

    row_means = clipped_df.mean(axis=1, skipna=True)

    df_aligned_fillna = clipped_df.mask(mask, row_means, axis=0)
    
    return df_aligned_fillna

def fillna_aligned_with_label(f:pd.DataFrame, label:pd.DataFrame):
    df = f.reindex(index = label.index, columns=f.columns.union(label.columns))
    df = df.apply(adjust_row, axis=1)
    # df = df.where(label.notna())
    return df


def adjust_row(row,
               low = 0.05,
               high = 0.95
               ):
    '''
        用每行的均值（裁剪百分之5， 百分之95分位数）填充每行的Nan值。
    '''
    q05 = row.quantile(low, interpolation='lower')
    q95 = row.quantile(high, interpolation='higher')
    truncated_row = row.clip(lower=q05, upper=q95)

    corrected_mean = truncated_row.mean(skipna=True)

    return row.fillna(corrected_mean)


# def clip_row(row):
#     '''
#         对每行进行裁剪百分之5、百分之95分位数的操作，不进行Nan值填充。
#     '''
#     q05 = row.quantile(0.05, interpolation='lower')  # 使用向下插值避免极端值被低估
#     q95 = row.quantile(0.95, interpolation='higher') # 使用向上插值避免极端值被高估
 
#     clipped_row = row.clip(lower=q05, upper=q95)
#     return clipped_row

def split_by_datetime_index(df:pd.DataFrame, 
                            inner_start = '2016-05-01',
                            outer_start = '2021-01-01',
                            outer_end = '2022-03-19'
                            ):
    '''
        Params:
            df: 二维, index只有DATETIME的DataFrame.
        Returns
        ---
        inner,outer
            内样本, 外样本。
    '''
    inner = df[(df.index >= inner_start) & (df.index < outer_start)]
    outer = df[(df.index >= outer_start) & (df.index < outer_end)]
    return inner, outer


def minute_align_fix_with_market(factor_value:pd.DataFrame, marketcap:pd.DataFrame):
    '''
        因子值与marketcap对齐并填均(裁剪)值然后mask.
        Params:
            factor_value: 待对齐的因子值
            marketcap: 待对齐的marketcap
        Returns
        ---
        factor_value
            对齐后的因子值
    '''
    common_cols = factor_value.columns.union(marketcap.columns)
    factor_value = factor_value.reindex(columns=common_cols)
    marketcap = marketcap.reindex(columns=common_cols)
    factor_value = factor_value.apply(adjust_row, axis=1)
    marketcap.index = pd.to_datetime(marketcap.index)
    factor_value.index = pd.to_datetime(factor_value.index)
    upsampled = marketcap.reindex(factor_value.index)
    factor_value = factor_value.where(upsampled>0)

    return factor_value

def split_by_int_index(df:pd.DataFrame, start:int, end:int):
    '''
        Params:
            df: 二维, index只有int的DataFrame.
        Returns
        ---
        result
    '''
    result = df[(df.index >= start) & (df.index < end)]
    return result

def generate_monthly_periods(start_year=2019, start_month=1, end_year=2025, end_month=4):
    periods = []
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        start_date = year * 10000 + month * 100 + 1
        if month == 12:
            end_date = (year + 1) * 10000 + 101
        else:
            end_date = year * 10000 + (month + 1) * 100 + 1
        period_name = f"{year}{month:02d}"
        periods.append({
            "start": start_date,
            "end": end_date,
            "path_suffix": period_name
        })
        if year == end_year and month == end_month:
            break
        month += 1
        if month > 12:
            month = 1
            year += 1
    return periods

if __name__ == '__main__':
    start = 20200101
    end = 20210101
    f = pd.read_parquet(r'D:\chenxing\Finforecast\factor_warehouse\factors_pricing_raw\factor1.parquet')
    result = split_by_int_index(f, start, end)
    print(result.head())
    print(result.tail())
    print(result.shape)
    print(f.shape)
    print(result.index)
    print(f.index)
    print(result.index.dtype)
    print(f.index.dtype)