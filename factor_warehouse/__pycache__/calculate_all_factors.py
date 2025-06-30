import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 获取当前文件所在路径
current_dir = Path(__file__).resolve().parent
# 添加到 sys.path
sys.path.append(str(current_dir))

from day_factor_zoo import *
from factor_validator import FactorValidator

def get_factor_descriptions():
    """获取所有因子的描述信息"""
    descriptions = {
        'F1': '转股溢价率: LAST(可转债价格/正股价格)/100*转股价格 - 1， 取得是最后一分钟',
        'F2': '修正转股溢价率：转股价值X，真实转股溢价率Y，修正转股溢价率Z = Y - Y_hat = Y - (a+b/X)',
        'F3': '纯债溢价率: LAST(可转债价格)/纯债价值 - 1， 取得是最后一分钟',
        'F4': '修正纯债溢价率：转股价值X，真实纯债溢价率Y，修正纯债溢价率Z = Y - Y_hat = Y - (aX + b)',
        'F5': '转股纯债溢价率: 转股价值/纯债价值 - 1',
        'F6': '双低因子: 转股溢价率 * 100 + 可转债价格',
        'F7': '修正双低因子： std(修正转股溢价率) + std(可转债价格)',
        'F8': '动量5日： 5日收益率',
        'F9': 'RSI: 5日stock涨幅之和 / 5日stock涨跌幅绝对值之和',
        'F10': 'Percent B: (stock收盘价 - 20日布林线下轨) / 20日布林带宽度',
        'F11': 'Price to High: stock收盘价 / 过去20日的stock最高价',
        'F12': 'Amihud: MEAN(abs(stock收盘价 - stock前收盘价) / stock成交量)',
        'F13': '日度资金流: ((stock_high + stock_low + stock_close) / 3) * stock_volume，5日资金流比率 = 5日正资金流之和 / 5日负资金流之和，5日MFI = 100 - 100 / (1 + 5日资金流比率)',
        'F14': '日内(转债与正股涨跌幅之差)的累计求和',
        'F15': '日间 转债与正股涨跌幅之差',
        'F16': '日内转债与正股涨跌幅的相关性',
        'F17': '日内温和收益：首先计算日内分钟数据对数收益率的中位数和 MAD，在中位数 1.96 倍 MAD 以内的分钟线定义为温和收益，将每日所有的温和对数收益率相加，得到温和收益',
        'F18': '分钟线方差均值：Mean(30分钟收益率方差)',
        'F19': '分钟线偏度均值：Mean(30分钟收益率偏度)',
        'F20': '分钟线均值方差：STD(30分钟收益率均值)',
        'F21': '分钟线偏度方差：STD(30分钟收益率偏度)',
        'F22': '分钟线方差方差：STD(30分钟收益率方差)',
        'F23': '收盘价与成交量相关系数',
        'F24': '早盘成交量占比',
        'F25': '尾盘成交量占比',
        'F26': '早盘收益率占比',
        'F27': '尾盘收益率占比',
        'F28': '日内收益率>0位置上的交易量占比',
        'F29': '已实现波动率：日内收益率平方之和',
        'F30': '（正收益率计算已实现波动率-负收益计算已实现波动率）/已实现波动率',
        'F31': '将当天分钟频率数据按照收益率正负划分，计算正收益 std 与负收益 std 差值',
        'F32': '计算日内1分钟收益率排序前后20%部分累计收益率 的 差 作为因子值',
        'F33': '计算除法和Log差值算法的两个1分钟收益率，每分钟计算2*(除法收益率-Log收益率)-Log收益率^2，并除以除法收益率绝对值全日均值，统计日内每分钟均值作为当天的因子值'
    }
    return descriptions

def calculate_factor_metrics(factor_df, label_df):
    """计算因子的IC、IR和缺失值比率"""
    try:
        # 计算IC
        IC = factor_df.T.corrwith(label_df.T)
        mean_ic = IC.mean()
        std_ic = IC.std(ddof=1)
        ir = mean_ic / std_ic if std_ic != 0 else np.nan
        
        # 计算缺失值比率
        missing_ratio = factor_df.isna().sum().sum() / factor_df.size
        
        return mean_ic, ir, missing_ratio
    except Exception as e:
        print(f"计算因子指标时出错: {e}")
        return np.nan, np.nan, np.nan

def main():
    """主函数：计算所有因子并生成文件"""
    print("开始计算所有因子...")
    
    # 创建factors目录（如果不存在）
    factors_dir = Path("D:/chenxing/Finforecast/factor_warehouse/factors")
    factors_dir.mkdir(exist_ok=True)
    
    # 加载label数据
    print("加载label数据...")
    label_path = "D:/chenxing/Finforecast/factor_warehouse/label/label_vwap_log_cliped"
    label = pd.read_parquet(label_path)
    print(f"Label数据形状: {label.shape}")
    
    # 获取因子描述
    factor_descriptions = get_factor_descriptions()
    
    # 定义所有因子函数
    factor_functions = {
        'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4, 'F5': F5,
        'F6': F6, 'F7': F7, 'F8': F8, 'F9': F9, 'F10': F10,
        'F11': F11, 'F12': F12, 'F13': F13, 'F14': F14, 'F15': F15,
        'F16': F16, 'F17': F17, 'F18': F18, 'F19': F19, 'F20': F20,
        'F21': F21, 'F22': F22, 'F23': F23, 'F24': F24, 'F25': F25,
        'F26': F26, 'F27': F27, 'F28': F28, 'F29': F29, 'F30': F30,
        'F31': F31, 'F32': F32, 'F33': F33
    }
    
    # 存储因子信息
    factor_info = []
    
    # 计算每个因子
    for factor_name, factor_func in factor_functions.items():
        print(f"\n正在计算 {factor_name}...")
        start_time = time.time()
        
        try:
            # 计算因子
            factor_df = factor_func()
            
            # 保存为parquet文件
            parquet_path = factors_dir / f"{factor_name}.parquet"
            factor_df.to_parquet(parquet_path)
            print(f"✓ {factor_name} 已保存到 {parquet_path}")
            
            # 计算指标
            ic, ir, missing_ratio = calculate_factor_metrics(factor_df, label)
            
            # 记录信息
            factor_info.append({
                '函数名称': factor_name,
                '因子表达式': factor_descriptions.get(factor_name, ''),
                'IC': ic,
                'IR': ir,
                '缺失值比率': missing_ratio
            })
            
            elapsed_time = time.time() - start_time
            print(f"✓ {factor_name} 计算完成，耗时 {elapsed_time:.2f} 秒")
            print(f"  - IC: {ic:.4f}, IR: {ir:.4f}, 缺失值比率: {missing_ratio:.4f}")
            
        except Exception as e:
            print(f"❌ {factor_name} 计算失败: {e}")
            # 记录错误信息
            factor_info.append({
                '函数名称': factor_name,
                '因子名称': factor_name,
                '因子表达式': factor_descriptions.get(factor_name, ''),
                'IC': np.nan,
                'IR': np.nan,
                '缺失值比率': np.nan
            })
    
    # 生成因子信息汇总CSV文件
    print("\n生成因子信息汇总...")
    factor_summary = pd.DataFrame(factor_info)
    csv_path = factors_dir / "factor_summary.csv"
    factor_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 因子信息汇总已保存到 {csv_path}")
    
    # 打印汇总信息
    print("\n=== 因子计算完成 ===")
    print(f"总共计算了 {len(factor_functions)} 个因子")
    print(f"成功计算: {len([f for f in factor_info if not pd.isna(f['IC'])])} 个")
    print(f"失败: {len([f for f in factor_info if pd.isna(f['IC'])])} 个")
    
    # 显示前10个因子的IC和IR
    print("\n前10个因子的IC和IR:")
    successful_factors = factor_summary[factor_summary['IC'].notna()].head(10)
    for _, row in successful_factors.iterrows():
        print(f"{row['函数名称']}: IC={row['IC']:.4f}, IR={row['IR']:.4f}")

if __name__ == "__main__":
    main() 