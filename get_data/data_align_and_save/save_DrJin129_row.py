import pandas as pd
import numpy as np
from tqdm import tqdm
from get_data.data_align_and_save import *

date_index = np.load("/home/hongkou/index_tradingday_201420250228.npy")
len_d = len(date_index)
print(len_d)
factor_colunms = np.load("/home/hongkou/index_Ticker_201420200604.npy")
print(len(factor_colunms))


dict_f = pd.read_pickle("/home/hongkou/dict_batch_features_new.pickle")
i=1

for key in tqdm(dict_f):
    factor = dict_f[key].squeeze()
    factor_df = pd.DataFrame(factor)
    factor_df.columns = factor_colunms
    factor_df_all = factor_df[:len_d]
    factor_df_all.index = date_index
    f = factor_df_all
    f.index = pd.to_datetime(f.index, format="%Y-%m-%d")
    # f.to_parquet(f"/home/hongkou/chenx/data_warehouse/DrJin_factors/row_half/half_F{i}.parquet")
    f.to_parquet(f"/home/USB_DRIVE1/Chenx_datawarehouse/DrJin/allday_row_factor/all_F{i}.parquet")
    assert f.shape == (len_d, len(factor_colunms))
    i += 1
