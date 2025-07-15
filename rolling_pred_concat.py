import pandas as pd
import os
from pathlib import Path

base_path = Path('D:/chenxing/Finforecast/exp')
path_list = [base_path / 'AttGRU_112_202401_202403/pred_csv/0.5GRU & 0.5GBDT_fin_pred_mask.csv',
             base_path / 'AttGRU_112_202404_202406/pred_csv/0.5GRU & 0.5GBDT_fin_pred_mask.csv',
             base_path / 'AttGRU_112_202407_202409/pred_csv/0.5GRU & 0.5GBDT_fin_pred_mask.csv',
             base_path / 'AttGRU_112_202410_202412/pred_csv/0.5GRU & 0.5GBDT_fin_pred_mask.csv',
             base_path / 'AttGRU_112_202501_202504/pred_csv/0.5GRU & 0.5GBDT_fin_pred_mask.csv',]

pred_concat_df = pd.DataFrame()

for path in path_list:
    pred_df = pd.read_csv(path)
    pred_concat_df = pd.concat([pred_concat_df, pred_df], axis=0)

pred_concat_df.to_csv('D:/chenxing/Finforecast/exp/rolling_pred/AttGRU_112_202401_202504.csv', index=False)

# first_col = pred_concat_df.columns[0]
# # 验证数据是否按时间正确排序
# print("=== 数据排序验证 ===")

# # 检查时间列的数据类型和格式
# print(f"时间列 '{first_col}' 的数据类型: {pred_concat_df[first_col].dtype}")
# print(f"时间列前5个值: {pred_concat_df[first_col].head().tolist()}")
# print(f"时间列后5个值: {pred_concat_df[first_col].tail().tolist()}")

# # 验证排序是否正确
# is_sorted = pred_concat_df[first_col].is_monotonic_increasing
# print(f"数据是否按时间升序排列: {is_sorted}")

# # 检查是否有重复的时间戳
# duplicate_times = pred_concat_df[first_col].duplicated().sum()
# print(f"重复时间戳数量: {duplicate_times}")

# # 显示时间范围
# min_time = pred_concat_df[first_col].min()
# max_time = pred_concat_df[first_col].max()
# print(f"时间范围: {min_time} 到 {max_time}")

# # 检查相邻行的时间差，确保没有时间倒流
# time_diffs = pred_concat_df[first_col].diff().dropna()
# negative_diffs = (time_diffs < 0).sum()
# print(f"时间倒流的行数: {negative_diffs}")

# if negative_diffs > 0:
#     print("⚠️ 警告：发现时间倒流的情况！")
#     # 显示时间倒流的具体位置
#     time_diffs_negative = time_diffs[time_diffs < 0]
#     print("时间倒流的位置:")
#     for idx, diff in time_diffs_negative.items():
#         try:
#             print(f"  位置 {idx}: 从 {pred_concat_df.loc[idx-1, first_col]} 到 {pred_concat_df.loc[idx, first_col]} (差值: {diff})")
#         except KeyError:
#             # 如果是第一个索引，无法访问前一行
#             print(f"  位置 {idx}: 第一个时间戳 {pred_concat_df.loc[idx, first_col]}")
#         except IndexError:
#             # 如果是最后一个索引，无法访问后一行
#             print(f"  位置 {idx}: 最后一个时间戳 {pred_concat_df.loc[idx, first_col]}")






