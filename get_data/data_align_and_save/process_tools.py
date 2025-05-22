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
    upsampled = marketcap.reindex(factor_value.index.normalize())
    upsampled.index = factor_value.index
    factor_value = factor_value.where(upsampled>0)

    return factor_value

