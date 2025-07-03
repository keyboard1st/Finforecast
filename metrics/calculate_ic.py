import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def ic_between_timestep(pred, y):
    """timestep中IC计算函数"""
    pred_series = pd.Series(pred.squeeze().cpu().numpy())
    y_series = pd.Series(y.squeeze().cpu().numpy())

    return pred_series.corr(y_series)

def ic_between_arr(pred, y):
    '''
    计算横表arr相关性
    '''
    pred_df = pd.DataFrame(pred)
    y_df = pd.DataFrame(y)
    ic = pred_df.T.corrwith(y_df.T)
    return ic.mean()

def ic_between_arr_new(pred, y):
    '''
    当ynan过多时可以用这个
    '''
    pred_df = pd.DataFrame(pred)
    y_df    = pd.DataFrame(y)
    ic_list = []
    for idx in pred_df.index:
        row_p = pred_df.loc[idx]
        row_y = y_df.loc[idx]
        valid = row_p.notna() & row_y.notna()
        if valid.sum() > 1:
            ic = row_p[valid].corr(row_y[valid])
            ic_list.append(ic)
    return np.nanmean(ic_list)

def ic_between_models_plot(model_pred_dict, save_path):

    true = model_pred_dict['label']
    model_pred_dict.pop('label', None)

    for name, pred_arr in model_pred_dict.items():
        ic = ic_between_arr(pred_arr, true)
        print(f"{name} model TEST IC: {ic:.4f}")

    model_names = list(model_pred_dict.keys())
    n = len(model_names)
    ic_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            pred1 = model_pred_dict[model_names[i]]
            pred2 = model_pred_dict[model_names[j]]
            ic_matrix[i, j] = ic_between_arr(pred1, pred2)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(ic_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=45, ha="left")
    ax.set_yticklabels(model_names)

    for i in range(n):
        for j in range(n):
            text = f"{ic_matrix[i, j]:.2f}"
            ax.text(j, i, text, va='center', ha='center', color='black')

    plt.title("model IC Correlation Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ic_bar(ic_dict, save_path):
    # 拆分字典的键和值
    models = list(ic_dict.keys())
    values = list(ic_dict.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, values, color='skyblue')
    ax.set_ylabel('Model')
    ax.set_xlabel('IC Value')

    # 设置标题和标签
    ax.set_title('IC Value by Model')

    # 动态调整 y 轴范围，给标签留空间
    min_val = min(values)
    max_val = max(values)
    y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.1
    ax.set_ylim(min_val - y_margin, max_val + y_margin)

    for bar, value in zip(bars, values):
        width = bar.get_width()
        ha = 'left' if width >= 0 else 'right'
        offset = y_margin * 0.1
        ax.text(
            width + offset if width >= 0 else width - offset,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.3f}',
            va='center', ha=ha, fontsize=9
        )

    # 显示图表
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def calculate_ic_metrics(returns_df, fin_factor, logger=None):
    """
    计算各种IC指标，返回字典
    """
    # 基础IC
    corr = returns_df.T.corrwith(fin_factor.T)
    if logger: logger.info(f"Final integrated IC: {corr.mean():.4f}")
    
    # 去除前N个时间步的IC
    mask3_corr = returns_df.iloc[3:,:].T.corrwith(fin_factor.iloc[3:,:].T)
    if logger: logger.info(f"Remove first 3 time steps IC: {mask3_corr.mean():.4f}")
    
    mask5_corr = returns_df.iloc[5:,:].T.corrwith(fin_factor.iloc[5:,:].T)
    if logger: logger.info(f"Remove first 5 time steps IC: {mask5_corr.mean():.4f}")
    
    mask15_corr = returns_df.iloc[15:,:].T.corrwith(fin_factor.iloc[15:,:].T)
    if logger: logger.info(f"Remove first 15 time steps IC: {mask15_corr.mean():.4f}")
    
    mask60_corr = returns_df.iloc[60:,:].T.corrwith(fin_factor.iloc[60:,:].T)
    if logger: logger.info(f"Remove first 60 time steps IC: {mask60_corr.mean():.4f}")
    
    return {
        "Final Concat IC": corr.mean(),
        "Mask3 IC": mask3_corr.mean(),
        "Mask5 IC": mask5_corr.mean(),
        "Mask15 IC": mask15_corr.mean(),
        "Mask60 IC": mask60_corr.mean()
    }



