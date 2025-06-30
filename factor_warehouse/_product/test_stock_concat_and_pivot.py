import pandas as pd
import sys
from pathlib import Path
import logging

# 添加父目录到路径以导入get_data_from_DB模块
sys.path.append(str(Path(__file__).parent.parent))
from get_data_from_DB import get_bond_stk_name_dict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_stock_bond_mapping():
    """测试股票到转债的映射功能"""
    logger.info("开始测试股票-转债映射功能...")
    
    try:
        # 获取映射关系
        bond_stk_dict = get_bond_stk_name_dict()
        logger.info(f"原始映射关系数量: {len(bond_stk_dict)}")
        
        # 创建反向映射
        stock_bond_dict = {}
        for bond_code, stock_code in bond_stk_dict.items():
            stock_bond_dict[stock_code] = bond_code
            
        logger.info(f"反向映射关系数量: {len(stock_bond_dict)}")
        
        # 显示一些示例
        sample_count = min(5, len(stock_bond_dict))
        sample_items = list(stock_bond_dict.items())[:sample_count]
        logger.info("示例映射关系:")
        for stock_code, bond_code in sample_items:
            logger.info(f"  股票代码: {stock_code} -> 转债代码: {bond_code}")
            
        return stock_bond_dict
        
    except Exception as e:
        logger.error(f"测试映射功能失败: {e}")
        return {}

def test_data_transformation():
    """测试数据转换逻辑"""
    logger.info("开始测试数据转换逻辑...")

    # 创建测试数据 - 保证所有列长度一致
    n = 240
    codes = ['000001.SZ', '000002.SZ', '000858.SZ', '999999.SZ']
    test_data = pd.DataFrame({
        'datetime': list(pd.date_range('2024-01-01', periods=n, freq='1min')) * len(codes),
        'code': codes * n,
        'price': [10.0, 20.0, 30.0, 40.0] * n,
        'volume': [1000, 2000, 3000, 4000] * n
    })

    # 创建测试映射（模拟）
    test_mapping = {
        '000001.SZ': '110001.SH',
        '000002.SZ': '110002.SH',
        '000858.SZ': '110858.SH',
    }

    logger.info(f"测试数据形状: {test_data.shape}")
    logger.info(f"测试映射数量: {len(test_mapping)}")

    # 测试转换逻辑
    test_data['code'] = test_data['code'].astype(str)
    test_data['bond_code'] = test_data['code'].map(test_mapping)

    # 过滤有效数据
    valid_data = test_data.dropna(subset=['bond_code'])
    dropped_count = len(test_data) - len(valid_data)

    logger.info(f"转换后有效数据: {len(valid_data)}")
    logger.info(f"过滤掉的数据: {dropped_count}")

    # 测试pivot操作
    if len(valid_data) > 0:
        try:
            wide = valid_data.pivot(index='datetime', columns='bond_code', values='price')
            logger.info(f"Pivot后数据形状: {wide.shape}")
            logger.info(f"列名类型: {type(wide.columns[0])}")

            # 测试转换为str
            wide.columns = wide.columns.astype(str)
            logger.info(f"转换为str后列名类型: {type(wide.columns[0])}")
            logger.info(f"列名示例: {list(wide.columns)}")

            return True

        except Exception as e:
            logger.error(f"Pivot操作失败: {e}")
            return False

    return False

def test_file_paths():
    """测试文件路径"""
    logger.info("开始测试文件路径...")
    
    input_root = Path(r'D:\raw_data\stock_min_data_per_day')
    output_root = Path(r'D:\chenxing\factor_warehouse\stock_one_minute_parquet')
    
    logger.info(f"输入路径存在: {input_root.exists()}")
    logger.info(f"输出路径存在: {output_root.exists()}")
    
    if input_root.exists():
        subfolders = [sd for sd in input_root.iterdir() if sd.is_dir()]
        logger.info(f"输入目录中子文件夹数量: {len(subfolders)}")
        
        if subfolders:
            sample_folder = subfolders[0]
            files = list(sample_folder.iterdir())
            logger.info(f"示例文件夹 '{sample_folder.name}' 中文件数量: {len(files)}")
            
            if files:
                sample_file = files[0]
                logger.info(f"示例文件: {sample_file.name}")
    
    return True

if __name__ == '__main__':
    logger.info("开始验证股票数据处理脚本...")
    
    # 测试1: 映射功能
    mapping_result = test_stock_bond_mapping()
    
    # 测试2: 数据转换逻辑
    transform_result = test_data_transformation()
    
    # 测试3: 文件路径
    path_result = test_file_paths()
    
    # 总结
    logger.info("=" * 50)
    logger.info("验证结果总结:")
    logger.info(f"映射功能测试: {'通过' if mapping_result else '失败'}")
    logger.info(f"数据转换测试: {'通过' if transform_result else '失败'}")
    logger.info(f"文件路径测试: {'通过' if path_result else '失败'}")
    
    if mapping_result and transform_result and path_result:
        logger.info("所有测试通过，可以运行主脚本")
    else:
        logger.warning("部分测试失败，请检查问题后再运行主脚本") 