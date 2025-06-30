import pandas as pd
import numpy as np

def check_data_quality():
    """检查swap_share_price数据的数据质量"""
    print("开始检查swap_share_price数据质量...")
    
    # 读取数据
    df = pd.read_parquet('Finforecast/factor_warehouse/daily_parquet/swap_share_price.parquet')
    
    print(f"数据基本信息:")
    print(f"形状: {df.shape}")
    print(f"索引范围: {df.index.min()} 到 {df.index.max()}")
    print(f"列数: {len(df.columns)}")
    
    # 检查数据变化情况
    print(f"\n检查前10列的数据变化情况:")
    for i, col in enumerate(df.columns[:10]):
        series = df[col].dropna()
        if len(series) > 0:
            unique_vals = series.nunique()
            change_rate = unique_vals / len(series)
            print(f"列 {col}: 非空值 {len(series)}, 唯一值 {unique_vals}, 变化率 {change_rate:.2%}")
            
            # 如果变化率很低，显示具体数据
            if change_rate < 0.1:  # 变化率低于10%
                print(f"  ⚠ 警告: 列 {col} 变化率很低!")
                print(f"  前10个值: {series.head(10).tolist()}")
                print(f"  唯一值: {sorted(series.unique())}")
    
    # 检查是否有完全不变的列
    print(f"\n检查完全不变的列:")
    static_columns = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) > 0 and series.nunique() == 1:
            static_columns.append(col)
            print(f"列 {col}: 完全不变，值 = {series.iloc[0]}")
    
    print(f"完全不变的列数: {len(static_columns)}")
    
    # 检查变化率分布
    print(f"\n变化率分布统计:")
    change_rates = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) > 0:
            change_rate = series.nunique() / len(series)
            change_rates.append(change_rate)
    
    change_rates = np.array(change_rates)
    print(f"平均变化率: {change_rates.mean():.2%}")
    print(f"中位数变化率: {np.median(change_rates):.2%}")
    print(f"最小变化率: {change_rates.min():.2%}")
    print(f"最大变化率: {change_rates.max():.2%}")
    
    # 检查低变化率的列
    low_change_cols = df.columns[change_rates < 0.1]
    print(f"\n变化率低于10%的列数: {len(low_change_cols)}")
    if len(low_change_cols) > 0:
        print("前10个低变化率列:")
        for col in low_change_cols[:10]:
            series = df[col].dropna()
            change_rate = series.nunique() / len(series)
            print(f"  {col}: 变化率 {change_rate:.2%}, 唯一值 {series.nunique()}")
    
    # 检查数据的时间连续性
    print(f"\n检查时间连续性:")
    df_sorted = df.sort_index()
    date_diff = df_sorted.index.to_series().diff()
    print(f"日期间隔统计:")
    print(f"  最小间隔: {date_diff.min()}")
    print(f"  最大间隔: {date_diff.max()}")
    print(f"  平均间隔: {date_diff.mean()}")
    
    # 检查是否有异常大的间隔
    large_gaps = date_diff[date_diff > 7]  # 间隔大于7天
    if len(large_gaps) > 0:
        print(f"发现 {len(large_gaps)} 个大于7天的间隔")
        print("前5个大间隔:")
        for date, gap in large_gaps.head().items():
            print(f"  {date}: 间隔 {gap} 天")

if __name__ == '__main__':
    check_data_quality() 