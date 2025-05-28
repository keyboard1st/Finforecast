import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def rename_and_copy_files(time_period: str):
    folder_path = f"/home/USB_DRIVE1/Chenx_datawarehouse/10min_factors/r_market_align_factor/{time_period}/"
    dst_folder = Path(f"/home/USB_DRIVE1/Chenx_datawarehouse/10min_factors/factor_rename/{time_period}/")
    mapping_file = dst_folder / "name_mapping.csv"

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                  if os.path.isfile(os.path.join(folder_path, f))]

    dst_folder.mkdir(parents=True, exist_ok=True)
    prefix_map = {}
    counter = 1
    records = []

    for file_path in tqdm(sorted(file_paths), desc=f"Processing {time_period}"):
        file_path = Path(file_path)
        parts = file_path.stem.split("_mkt_")
        if len(parts) != 2:
            print(f"Unexpected filename format: {file_path.name}")
            continue

        factors_name = parts[0]
        sample_set = parts[1].split(".")[0]

        if factors_name not in prefix_map:
            prefix_map[factors_name] = f"F{counter}"
            counter += 1

        new_name = f"{prefix_map[factors_name]}_{sample_set}.parquet"
        dst_path = dst_folder / new_name
        shutil.copy(file_path, dst_path)

        records.append({
            "new_name": prefix_map[factors_name],
            "factors_original_name": factors_name
        })

    if records:
        df = pd.DataFrame(records).drop_duplicates()
        df.to_csv(mapping_file, index=False)
        print(f"Mapping table saved to: {mapping_file}")

time_periods = ["2021-2022", "2022-2023", "2023-2024"]
for period in time_periods:
    print(f"Processing {period}...")
    rename_and_copy_files(period)



