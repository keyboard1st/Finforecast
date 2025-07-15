#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时间解析工具
功能：将时间段字符串拆分成月份列表、年度范围等
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


def parse_period_to_months(time_period):
    """
    将时间段字符串解析为月份列表
    
    Args:
        time_period: str, 支持以下格式:
            - '202107-202206': 年月格式，从2021年7月到2022年6月
            - '2021-2022': 年份格式，从2021年1月到2022年12月
    
    Returns:
        list: 月份字符串列表，如 ['202107', '202108', ..., '202206']
    
    Examples:
        >>> parse_period_to_months('202107-202206')
        ['202107', '202108', '202109', '202110', '202111', '202112', 
         '202201', '202202', '202203', '202204', '202205', '202206']
        
        >>> parse_period_to_months('2021-2021')
        ['202101', '202102', ..., '202112']
    """
    start_str, end_str = time_period.split('-')
    
    if len(start_str) == 6 and len(end_str) == 6:
        # 年月格式: 202107-202206
        start_year = int(start_str[:4])
        start_month = int(start_str[4:])
        end_year = int(end_str[:4])
        end_month = int(end_str[4:])
        
        months = []
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        
        while current_date <= end_date:
            months.append(f"{current_date.year:04d}{current_date.month:02d}")
            current_date += relativedelta(months=1)
        
        return months
    
    elif len(start_str) == 4 and len(end_str) == 4:
        # 年份格式: 2021-2022
        start_year = int(start_str)
        end_year = int(end_str)
        
        months = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                months.append(f"{year:04d}{month:02d}")
        
        return months
    
    else:
        raise ValueError(f"不支持的时间格式: {time_period}，请使用 'YYYYMM-YYYYMM' 或 'YYYY-YYYY' 格式")


def parse_period_to_year_ranges(time_period):
    """
    将时间段字符串解析为年度范围列表（适配现有文件夹结构）
    
    Args:
        time_period: str, 格式如 '202107-202206' 或 '2021-2022'
    
    Returns:
        list: 年度范围字符串列表，如 ['2021-2022', '2022-2023']
    
    Examples:
        >>> parse_period_to_year_ranges('202107-202206')
        ['2021-2022']
        
        >>> parse_period_to_year_ranges('2020-2022')
        ['2020-2021', '2021-2022', '2022-2023']
    """
    start_str, end_str = time_period.split('-')
    
    if len(start_str) == 6 and len(end_str) == 6:
        # 年月格式: 202107-202206
        start_year = int(start_str[:4])
        end_year = int(end_str[:4])
    elif len(start_str) == 4 and len(end_str) == 4:
        # 年份格式: 2021-2022
        start_year = int(start_str)
        end_year = int(end_str)
    else:
        raise ValueError(f"不支持的时间格式: {time_period}")
    
    year_ranges = []
    for year in range(start_year, end_year + 1):
        year_ranges.append(f"{year}-{year+1}")
    
    return year_ranges


def parse_period_to_quarters(time_period):
    """
    将时间段字符串解析为季度列表
    
    Args:
        time_period: str, 格式如 '202107-202206' 或 '2021-2022'
    
    Returns:
        list: 季度字符串列表，如 ['2021Q3', '2021Q4', '2022Q1', '2022Q2']
    """
    months = parse_period_to_months(time_period)
    quarters = []
    
    for month in months:
        year = int(month[:4])
        month_num = int(month[4:])
        quarter = (month_num - 1) // 3 + 1
        quarter_str = f"{year}Q{quarter}"
        
        if quarter_str not in quarters:
            quarters.append(quarter_str)
    
    return quarters


def create_month_date_range(time_period):
    """
    创建月度日期范围的DataFrame索引
    
    Args:
        time_period: str, 格式如 '202107-202206'
    
    Returns:
        pd.DatetimeIndex: 月度日期索引
    """
    months = parse_period_to_months(time_period)
    dates = []
    
    for month in months:
        year = int(month[:4])
        month_num = int(month[4:])
        dates.append(datetime(year, month_num, 1))
    
    return pd.DatetimeIndex(dates)


def demo_time_parsing():
    """
    演示所有时间解析功能
    """
    print("="*80)
    print("时间字符串解析工具演示")
    print("="*80)
    
    test_cases = [
        '202107-202206',  # 年月格式，跨年
        '202101-202112',  # 年月格式，单年
        '2021-2022',      # 年份格式，2年
        '2020-2020',      # 年份格式，单年
    ]
    
    for time_period in test_cases:
        print(f"\n📅 输入时间段: {time_period}")
        print("-" * 60)
        
        try:
            # 1. 解析为月份列表
            months = parse_period_to_months(time_period)
            print(f"🗓️  月份列表 ({len(months)}个月):")
            if len(months) <= 12:
                print(f"   {months}")
            else:
                print(f"   开始: {months[:3]}")
                print(f"   结束: {months[-3:]}")
                print(f"   (省略中间{len(months)-6}个月)")
            
            # 2. 解析为年度范围
            year_ranges = parse_period_to_year_ranges(time_period)
            print(f"📊 年度范围 ({len(year_ranges)}个): {year_ranges}")
            
            # 3. 解析为季度
            quarters = parse_period_to_quarters(time_period)
            print(f"📈 季度列表 ({len(quarters)}个): {quarters}")
            
            # 4. 创建日期索引
            date_index = create_month_date_range(time_period)
            print(f"📆 日期索引: {date_index[0].strftime('%Y-%m')} 到 {date_index[-1].strftime('%Y-%m')}")
            
        except Exception as e:
            print(f"❌ 解析错误: {e}")


if __name__ == '__main__':
    demo_time_parsing()
    
    print("\n" + "="*80)
    print("快速使用示例")
    print("="*80)
    
    # 快速使用示例
    time_period = '202107-202206'
    
    print(f"🎯 处理时间段: {time_period}")
    months = parse_period_to_months(time_period)
    print(f"   包含月份: {len(months)} 个")
    print(f"   起始月份: {months[0]}")
    print(f"   结束月份: {months[-1]}")
    
    # 遍历每个月
    print(f"\n🔄 遍历每个月:")
    for i, month in enumerate(months):
        if i < 3 or i >= len(months) - 3:
            print(f"   月份 {i+1:2d}: {month}")
        elif i == 3:
            print(f"   ... (省略 {len(months)-6} 个月)") 