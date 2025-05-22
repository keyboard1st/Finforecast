import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

def backtest_strategy(pred_returns, labels, market_cap):
    # mktcap_align = market_cap.reindex(index=pred_returns.index, columns=pred_returns.columns)
    labels_align = labels.reindex(index=pred_returns.index, columns=pred_returns.columns)
    pred_returns = pred_returns.where(market_cap>0)
    # 1. Calculate daily weights
    weights = pd.DataFrame(0, index=pred_returns.index, columns=pred_returns.columns)
    # mask = (pred_returns.sub(pred_returns.quantile(0.9, axis=1), axis=0) >= 0) & (mktcap_align > 0)
    mask = (pred_returns.sub(pred_returns.quantile(0.6, axis=1), axis=0) >= 0)
    labels_selected = labels_align.where(mask)
    # weights = weights.where(pred_returns.sub(pred_returns.quantile(0.9, axis=1), axis=0) >= 0).add(1/labels_selected.count(axis=1), axis=0)
    pred_selected = pred_returns.where(mask)
    weights = weights.where(mask).add( 1 /pred_selected.count(axis=1), axis=0)
    weights = weights.replace(np.nan, 0)
    # 2. Calculate daily portfolio returns
    daily_pnl = (weights * labels_selected).sum(axis=1)
    cumulative_returns = (1 + daily_pnl).cumprod() - 1

    # 3. Calculate metrics
    # Max Drawdown
    max_drawdown = (cumulative_returns.expanding().max() - cumulative_returns).max()

    # Annualized Return
    annualized_return = (1 + cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1

    # Sharpe Ratio
    sharpe_ratio = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)

    # Turnover Rate (双边换手率)
    turnover_series = (weights.diff().abs().sum(axis=1) / 2)


    return weights, {
        'cumulative_returns': cumulative_returns,
        'max_drawdown': max_drawdown,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'turnover_series' :turnover_series
    }

def plot_model_metrics_and_save(pred_df_list, save_path):
    for model_name, (weights, metrics) in pred_df_list.items():
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Performance: {model_name}', fontsize=16)

        # 1. Cumulative Returns
        metrics['cumulative_returns'].plot(ax=axes[0, 0], title='Cumulative Returns')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True)

        # 2. Turnover Rate Time Series
        metrics['turnover_series'].plot(ax=axes[0, 1], title='Daily Turnover Rate')
        axes[0, 1].set_ylabel('Turnover Rate')
        axes[0, 1].axhline(y=metrics['turnover_series'].mean(), color='r', linestyle='--',
                           label=f'Mean: {metrics["turnover_series"].mean():.2%}')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. Key Metrics Table
        metric_text = (
            f"Annualized Return: {metrics['annualized_return']:.2%}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Avg Turnover: {metrics['turnover_series'].mean():.2%}"
        )
        axes[1, 0].axis('off')
        axes[1, 0].text(0.1, 0.5, 'Key Metrics:\n\n' + metric_text, fontsize=12, verticalalignment='center')

        # 4. Weight Distribution Heatmap (Optional)
        if not weights.empty:
            weight_distribution = weights.mean().sort_values(ascending=False).head(10)
            weight_distribution.plot(kind='barh', ax=axes[1, 1], title='Top Asset Average Weights')
            axes[1, 1].set_xlabel('Average Weight')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path,f'{model_name}_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()

def plot_model_metrics(pred_df_list):
    for model_name, (weights, metrics) in pred_df_list.items():
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Performance: {model_name}', fontsize=16)

        # 1. Cumulative Returns
        metrics['cumulative_returns'].plot(ax=axes[0, 0], title='Cumulative Returns')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True)

        # 2. Turnover Rate Time Series
        metrics['turnover_series'].plot(ax=axes[0, 1], title='Daily Turnover Rate')
        axes[0, 1].set_ylabel('Turnover Rate')
        axes[0, 1].axhline(y=metrics['turnover_series'].mean(), color='r', linestyle='--',
                           label=f'Mean: {metrics["turnover_series"].mean():.2%}')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. Key Metrics Table
        metric_text = (
            f"Annualized Return: {metrics['annualized_return']:.2%}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Avg Turnover: {metrics['turnover_series'].mean():.2%}"
        )
        axes[1, 0].axis('off')
        axes[1, 0].text(0.1, 0.5, 'Key Metrics:\n\n' + metric_text, fontsize=12, verticalalignment='center')

        # 4. Weight Distribution Heatmap (Optional)
        if not weights.empty:
            weight_distribution = weights.mean().sort_values(ascending=False).head(10)
            weight_distribution.plot(kind='barh', ax=axes[1, 1], title='Top Asset Average Weights')
            axes[1, 1].set_xlabel('Average Weight')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    exp_path = '/home/hongkou/TimeSeries/exp/CY_2022_2023'
    time_period = '2022-2023'
    pred_path_list = [(os.path.join(exp_path,'pred_csv/GRU_fin_pred_mask.csv'), 'GRU'), (os.path.join(exp_path,'pred_csv/LightGBM & XGBoost & CatBoost_fin_pred_mask.csv'), 'LGBT'), (os.path.join(exp_path,'pred_csv/GRU & LightGBM & CatBoost & XgBoost_fin_pred_mask.csv'), 'Mean')]
    labels = pd.read_parquet(f'/home/USB_DRIVE1/Chenx_datawarehouse/CY/rolling_factors/r_label/{time_period}/label_outer.parquet')
    market_cap = pd.read_parquet(r"/home/hongkou/chenx/data_warehouse/marketcap.parquet")
    yao_full_res = {}
    for p, name in tqdm(pred_path_list):
        yao_full_res[name] = backtest_strategy(pd.read_csv(p).set_index('DATETIME').astype(np.float32), labels, market_cap)

    # Usage example
    plot_model_metrics(yao_full_res)
