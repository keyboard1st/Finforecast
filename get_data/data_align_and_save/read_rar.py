import os
import rarfile

# RAR 多卷时只提供第三卷，rarfile 会尽量从这一卷提取
rar_path = '/home/USB_DRIVE1/Data.part3.rar'
out_dir  = '/home/USB_DRIVE1/Chenx_datawarehouse/CY/row_factors/'
os.makedirs(out_dir, exist_ok=True)

# 打开 RAR（rarfile 自动回落到 unrar 工具）
with rarfile.RarFile(rar_path) as rf:
    # 遍历所有文件条目
    for info in rf.infolist():
        # 只处理 .zst 文件
        if not info.filename.lower().endswith('.zst'):
            continue

        try:
            # 直接读取整个文件数据（如果跨卷缺损，会抛异常）
            data = rf.read(info)
        except Exception as e:
            print(f"❌ 无法完整提取 {info.filename}: {e}")
            continue

        # 目标路径，保留原始文件名
        # 如果 info.filename 带了子目录，这里只取最后的 basename
        fname = os.path.basename(info.filename)
        dst = os.path.join(out_dir, fname)

        # 写出
        with open(dst, 'wb') as f:
            f.write(data)

        print(f"✅ 提取 {info.filename} → {dst}")

