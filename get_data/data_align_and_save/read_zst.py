import os
import zstandard as zstd

src_dir = "/home/USB_DRIVE1/Chenx_datawarehouse/CY/row_factors_zst"
dst_dir = "/home/USB_DRIVE1/Chenx_datawarehouse/CY/row_factors"

os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if fname.endswith(".zst"):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname[:-4])  # 去掉 .zst

        with open(src_path, 'rb') as compressed:
            with open(dst_path, 'wb') as decompressed:
                dctx = zstd.ZstdDecompressor()
                dctx.copy_stream(compressed, decompressed)

        print(f"Decompressed: {fname} → {dst_path}")