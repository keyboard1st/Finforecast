import torch


def filter_and_fillna(batch_x, batch_y, threshold=0.8):
    valid_mask = ~torch.isnan(batch_y.squeeze())

    batch_x = batch_x[valid_mask]
    batch_y = batch_y[valid_mask]

    batch_x = torch.nan_to_num(batch_x, nan=0)

    return batch_x, batch_y

def dropx_and_fillna(batch_x, batch_y, threshold=0.8):
    valid_mask = ~torch.isnan(batch_y.squeeze())
    nan_counts = torch.isnan(batch_x).sum(dim=(1, 2))  # 每个样本的NaN数量，shape=(B,)
    total_elements = batch_x.shape[1] * batch_x.shape[2]  # 每个样本总元素数 T*C
    nan_mask = (nan_counts.float() / total_elements) <= threshold  # NaN比例≤80%的样本掩码，shape=(B,)

    combined_mask = valid_mask & nan_mask

    batch_x = batch_x[combined_mask]
    batch_y = batch_y[combined_mask]

    batch_x = torch.nan_to_num(batch_x, nan=0)

    return batch_x, batch_y