import os
from pathlib import Path

# 设置目标根目录
one_minute_parquet_dir = Path(r'D:\chenxing\Finforecast\factor_warehouse\one_minute_parquet')

# 检查目录是否存在
if not one_minute_parquet_dir.exists():
    print(f"目录不存在: {one_minute_parquet_dir}")
    exit(1)

# 统计删除的文件数量
deleted_files = []

# 遍历所有子文件夹
for subfolder in one_minute_parquet_dir.iterdir():
    if subfolder.is_dir():
        # 查找以stock开头的parquet文件
        for file in subfolder.glob('stock_*.parquet'):
            try:
                file.unlink()
                deleted_files.append(str(file))
                print(f"已删除: {file}")
            except Exception as e:
                print(f"删除 {file} 时出错: {e}")

print(f"\n共删除 {len(deleted_files)} 个文件。") 