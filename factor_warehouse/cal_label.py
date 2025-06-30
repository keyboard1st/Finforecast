import pandas as pd
import numpy as np
from get_data_from_pq import get_minute_ccb_all, get_date_list


def vwap():
    '''
    14：30-14：40的成交量加权平均价格
    '''
    dates = get_date_list()
    def cal_vwap(date):
        vol = get_minute_ccb_all(date,'volume')
        amt = get_minute_ccb_all(date,'total_turnover')
        t = vol.index.values % 1_000_000
        mask = (t > 143000) & (t <= 144000)
        vol_filtered = vol[mask].sum()
        amt_filtered = amt[mask].sum()
        # vol_masked = vol[mask]
        # amt_masked = amt[mask]
        # vol_zero_count = (vol_masked == 0).sum(axis=0)
        # amt_zero_count = (amt_masked == 0).sum(axis=0)

        # zero_too_many = (vol_zero_count >= 8) | (amt_zero_count >= 8)
        # vol_filtered = vol_masked.sum()
        # amt_filtered = amt_masked.sum()
        vwap = amt_filtered/vol_filtered
        return vwap
    series_dict = { date: cal_vwap(date) for date in dates }
    df = pd.concat(series_dict, axis=1).T
    return df

def label():
    df = vwap()
    # daily_ret = df.pct_change(fill_method=None).shift(-1)

    log_ret = np.log(df.shift(-1) / df)
    # daily_ret.replace([np.inf, -np.inf], np.nan, inplace=True)
    log_ret.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 对log_ret进行winsorization操作，对每行使用95%和5%的分位数替代异常值
    def winsorize_row(row):
        non_null_count = row.notna().sum()
        if non_null_count >= 10:
            row_clean = row.dropna()
            lower_bound = row_clean.quantile(0.05)
            upper_bound = row_clean.quantile(0.95)
            # 确保分位数不是NaN
            if pd.notna(lower_bound) and pd.notna(upper_bound):
                return row.clip(lower=lower_bound, upper=upper_bound)
        return row
    
    log_ret = log_ret.apply(winsorize_row, axis=1)
    log_ret = log_ret.iloc[:-1]
    
    return log_ret

if __name__ == '__main__':
    # 生成新的label
    new_label = label()
    new_label.to_parquet('factor_warehouse/label_raw/label_vwap_log_cliped_new.parquet')
    print(f"\n新生成的label已保存到: factor_warehouse/label_raw/label_vwap_log_cliped_new.parquet")
    