import os
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import sys
import re
from pymongo import MongoClient
from datetime import datetime
import numpy as np

label = pd.read_parquet(r'D:\chenxing\Finforecast\factor_warehouse\label\label_vwap_log_cliped')
client = MongoClient(
        'mongodb://chenhang:AZchenhang123@192.168.1.106:27017/?authMechanism=SCRAM-SHA-1&directConnection=true')
database = client['cb_factor_enc']
collection = database['pricing1430']
docs = collection.find()
df = pd.DataFrame(list(docs))
df['date'] = df['date'].str.replace('-', '').astype(np.int32)
df['code'] = df['code'].astype(np.int32)

save_path = r'D:\chenxing\Finforecast\factor_warehouse\factors_pricing'

# for factor in tqdm(range(1, 84)):
#     factor_name = f'factor{factor}'
#     factor_df = df.pivot(index='date', columns='code', values=factor_name)
#     factor_df = factor_df.reindex(index=label.index, columns=label.columns)
#     factor_df.to_parquet(os.path.join(save_path, f'{factor_name}.parquet'))

factor_name = f'factor51'
factor_df = df.pivot(index='date', columns='code', values=factor_name)
factor_df = factor_df.replace({False: 0, True: 1})
factor_df = factor_df.reindex(index=label.index, columns=label.columns)
factor_df.to_parquet(os.path.join(save_path, f'{factor_name}.parquet'))

factor_name = f'factor52'
factor_df = df.pivot(index='date', columns='code', values=factor_name)
factor_df = factor_df.replace({'OHL': np.nan, 'OLL': np.nan, 'A-': 0, 'A': 1, 'A+': 2, 'AA-': 3, 'AA':4, 'AA+': 5, 'AAA': 6})
factor_df = factor_df.reindex(index=label.index, columns=label.columns)
factor_df.to_parquet(os.path.join(save_path, f'{factor_name}.parquet'))

factor_name = f'factor53'
factor_df = df.pivot(index='date', columns='code', values=factor_name)
factor_df = factor_df.replace({'OHL': np.nan, 'OLL': np.nan, 'A-': 0, 'A': 1, 'A+': 2, 'AA-': 3, 'AA':4, 'AA+': 5, 'AAA': 6})
factor_df = factor_df.reindex(index=label.index, columns=label.columns)
factor_df.to_parquet(os.path.join(save_path, f'{factor_name}.parquet'))