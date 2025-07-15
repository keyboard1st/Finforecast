"""
量化回测模块 - 优化版本

主要功能:
1. 策略回测计算 (支持多种选股策略)
2. 风险指标计算 (夏普比率、最大回撤、信息比率等)
3. 可视化分析
4. 策略对比分析

作者: 量化研究团队
版本: 2.0 (优化版)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from datetime import datetime
from tqdm import tqdm
import os
import warnings
from typing import Dict, Tuple, Optional, Union, List
import logging

# Configure matplotlib and logging
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtest Class - Integrated backtesting functionality
    """
    
    def __init__(self, trading_days_per_year: int = 252):
        """
        Initialize backtest engine
        
        Args:
            trading_days_per_year: Trading days per year, default 252
        """
        self.trading_days_per_year = trading_days_per_year
        self.results_cache = {}
        
    def validate_data(self, pred_returns: pd.DataFrame, labels: pd.DataFrame) -> bool:
        """
        Data validation function
        
        Args:
            pred_returns: Predicted returns matrix
            labels: True labels matrix
            
        Returns:
            bool: Whether data is valid
        """
        try:
            # Basic shape check
            if pred_returns.shape != labels.shape:
                logger.warning(f"Prediction data shape {pred_returns.shape} does not match label data shape {labels.shape}")
                return False
            
            # Check for empty data
            if pred_returns.empty or labels.empty:
                logger.error("Input data is empty")
                return False
            
            # Check data types
            if not (pred_returns.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))).all():
                logger.warning("Prediction data contains non-numeric types")
                
            if not (labels.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))).all():
                logger.warning("Label data contains non-numeric types")
            
            # Data quality statistics
            pred_nan_ratio = pred_returns.isna().sum().sum() / (pred_returns.shape[0] * pred_returns.shape[1])
            label_nan_ratio = labels.isna().sum().sum() / (labels.shape[0] * labels.shape[1])
            
            logger.info(f"Data validation passed: Prediction NaN ratio {pred_nan_ratio:.1%}, Label NaN ratio {label_nan_ratio:.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False

    def calculate_advanced_metrics(self, daily_pnl: pd.Series, cumulative_returns: pd.Series,
                                 benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate advanced risk metrics
        
        Args:
            daily_pnl: Daily P&L series
            cumulative_returns: Cumulative returns series
            benchmark_returns: Benchmark returns series (optional)
            
        Returns:
            Dict: Risk metrics dictionary
        """
        try:
            metrics = {}
            
            # Basic metrics
            metrics['total_return'] = cumulative_returns.iloc[-1]
            metrics['annualized_return'] = (1 + cumulative_returns.iloc[-1]) ** (self.trading_days_per_year / len(cumulative_returns)) - 1
            metrics['volatility'] = daily_pnl.std() * np.sqrt(self.trading_days_per_year)
            metrics['sharpe_ratio'] = daily_pnl.mean() / daily_pnl.std() * np.sqrt(self.trading_days_per_year) if daily_pnl.std() != 0 else 0
            
            # Drawdown analysis
            peak = cumulative_returns.expanding().max()
            drawdown = cumulative_returns - peak
            metrics['max_drawdown'] = abs(drawdown.min())
            
            # Calculate maximum drawdown duration
            dd_duration = self._calculate_drawdown_duration(drawdown)
            metrics['max_drawdown_duration'] = dd_duration
            
            # Calmar ratio (annualized return / max drawdown)
            metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] != 0 else 0
            
            # Win rate and profit/loss ratio
            positive_returns = daily_pnl[daily_pnl > 0]
            negative_returns = daily_pnl[daily_pnl < 0]
            
            metrics['win_rate'] = len(positive_returns) / len(daily_pnl)
            metrics['profit_loss_ratio'] = abs(positive_returns.mean() / negative_returns.mean()) if len(negative_returns) > 0 and negative_returns.mean() != 0 else np.inf
            
            # If benchmark exists, calculate relative metrics
            if benchmark_returns is not None:
                active_returns = daily_pnl - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(self.trading_days_per_year)
                metrics['information_ratio'] = active_returns.mean() / active_returns.std() * np.sqrt(self.trading_days_per_year) if active_returns.std() != 0 else 0
                metrics['tracking_error'] = tracking_error
                
            return metrics
            
        except Exception as e:
            logger.error(f"Advanced metrics calculation failed: {e}")
            return {}

    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """
        Calculate maximum drawdown duration
        
        Args:
            drawdown: Drawdown series
            
        Returns:
            int: Maximum drawdown duration (days)
        """
        try:
            current_duration = 0
            max_duration = 0
            
            for dd in drawdown:
                if dd < 0:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
                    
            return max_duration
            
        except Exception as e:
            logger.warning(f"Drawdown duration calculation failed: {e}")
            return 0

    def backtest_strategy(self, pred_returns: pd.DataFrame, labels: pd.DataFrame, 
                         market_cap: Optional[pd.DataFrame] = None,
                         top_pct: float = 0.1, 
                         weight_method: str = 'equal',
                         min_position: int = 1,
                         max_position: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Optimized strategy backtest function
        
        Args:
            pred_returns: Predicted returns matrix [time, stock_code]
            labels: True labels matrix [time, stock_code]
            market_cap: Market cap matrix (optional) [time, stock_code]
            top_pct: Top percentage selection, default 10%
            weight_method: Weighting method ('equal', 'value_weighted', 'risk_parity')
            min_position: Minimum position count
            max_position: Maximum position count (optional)
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (weight matrix, backtest metrics dict)
        """
        try:
            # Data validation
            if not self.validate_data(pred_returns, labels):
                raise ValueError("Data validation failed")
            
            logger.info(f"Starting backtest: Top {top_pct:.1%} stocks, weight method={weight_method}")
            
            # Data alignment
            labels_align = labels.reindex(index=pred_returns.index, columns=pred_returns.columns)
            nan_mask = labels_align.isna()
            pred_clean = pred_returns.where(~nan_mask)
            
            # Market cap filtering (if provided)
            if market_cap is not None:
                market_cap_align = market_cap.reindex(index=pred_returns.index, columns=pred_returns.columns)
                valid_cap_mask = market_cap_align > 0
                pred_clean = pred_clean.where(valid_cap_mask)
                logger.info("Market cap filtering applied")
            
            # Initialize weight matrix
            weights = pd.DataFrame(0.0, index=pred_clean.index, columns=pred_clean.columns)
            
            # Calculate daily stock selection and weights
            daily_metrics = []
            
            for date_idx in tqdm(range(len(pred_clean)), desc="Calculating daily weights"):
                date = pred_clean.index[date_idx]
                day_pred = pred_clean.iloc[date_idx]
                day_labels = labels_align.iloc[date_idx]
                
                # Select valid stocks
                valid_stocks = day_pred.dropna()
                
                if len(valid_stocks) == 0:
                    logger.warning(f"Date {date}: No valid prediction data")
                    continue
                
                # Calculate selection count
                select_num = max(min_position, min(int(len(valid_stocks) * top_pct), 
                                                 max_position or len(valid_stocks)))
                
                # Select top N stocks
                top_stocks = valid_stocks.nlargest(select_num)
                
                # Calculate weights
                if weight_method == 'equal':
                    # Equal weight
                    stock_weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
                elif weight_method == 'value_weighted' and market_cap is not None:
                    # Market cap weighted
                    day_cap = market_cap_align.iloc[date_idx]
                    cap_weights = day_cap[top_stocks.index].fillna(0)
                    cap_weights = cap_weights / cap_weights.sum() if cap_weights.sum() > 0 else pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
                    stock_weights = cap_weights
                elif weight_method == 'risk_parity':
                    # Risk parity (simplified version using inverse of prediction values)
                    risk_weights = 1 / (abs(top_stocks) + 1e-8)  # Avoid division by zero
                    stock_weights = risk_weights / risk_weights.sum()
                else:
                    # Default equal weight
                    stock_weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
                
                # Set weights
                weights.loc[date, stock_weights.index] = stock_weights.values
                
                # Record daily statistics
                daily_metrics.append({
                    'date': date,
                    'selected_count': len(top_stocks),
                    'valid_labels_count': day_labels[top_stocks.index].count(),
                    'avg_pred_score': top_stocks.mean(),
                    'pred_score_std': top_stocks.std()
                })
            
            # Calculate portfolio returns
            labels_selected = labels_align.where(weights > 0)
            daily_pnl = (weights * labels_selected).sum(axis=1)
            cumulative_returns = (1 + daily_pnl).cumprod() - 1
            
            # Calculate turnover rate
            turnover_series = (weights.diff().abs().sum(axis=1) / 2).fillna(0)
            
            # Calculate advanced metrics
            advanced_metrics = self.calculate_advanced_metrics(daily_pnl, cumulative_returns)
            
            # Organize results
            results = {
                'daily_pnl': daily_pnl,
                'cumulative_returns': cumulative_returns,
                'turnover_series': turnover_series,
                'daily_metrics': pd.DataFrame(daily_metrics),
                **advanced_metrics
            }
            
            # Add strategy configuration info
            results['strategy_config'] = {
                'top_pct': top_pct,
                'weight_method': weight_method,
                'min_position': min_position,
                'max_position': max_position,
                'total_trading_days': len(pred_clean),
                'avg_selected_stocks': np.mean([m['selected_count'] for m in daily_metrics])
            }
            
            logger.info(f"Backtest completed: Annualized return {advanced_metrics.get('annualized_return', 0):.2%}, "
                       f"Sharpe ratio {advanced_metrics.get('sharpe_ratio', 0):.3f}")
            
            return weights, results
            
        except Exception as e:
            logger.error(f"Backtest calculation failed: {e}")
            raise

    def compare_strategies(self, pred_returns: pd.DataFrame, labels: pd.DataFrame,
                          strategies_config: List[Dict]) -> Dict:
        """
        Strategy comparison analysis
        
        Args:
            pred_returns: Predicted returns matrix
            labels: True labels matrix
            strategies_config: Strategy configuration list
            
        Returns:
            Dict: Backtest results for each strategy
        """
        try:
            logger.info(f"Starting comparison of {len(strategies_config)} strategies")
            
            results = {}
            
            for i, config in enumerate(strategies_config):
                strategy_name = config.get('name', f'Strategy_{i+1}')
                logger.info(f"Backtesting strategy: {strategy_name}")
                
                # Remove name parameter, pass other parameters to backtest_strategy
                backtest_params = {k: v for k, v in config.items() if k != 'name'}
                
                weights, metrics = self.backtest_strategy(pred_returns, labels, **backtest_params)
                results[strategy_name] = {'weights': weights, 'metrics': metrics}
            
            return results
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            raise

    def create_enhanced_plots(self, results_dict: Dict, save_path: Optional[str] = None, 
                             pred_returns: Optional[pd.DataFrame] = None, labels: Optional[pd.DataFrame] = None):
        """
        Create enhanced visualization charts with decile and quintile analysis
        
        Args:
            results_dict: Backtest results dictionary
            save_path: Save path (optional)
            pred_returns: Predicted returns matrix (needed for group analysis)
            labels: True labels matrix (needed for group analysis)
        """
        try:
            # Flag to track if group analysis has been created
            group_analysis_created = False
            
            # Calculate market average returns if labels are provided
            market_returns = None
            market_cumulative_returns = None
            if labels is not None:
                # Calculate daily market average returns (mean of each row)
                market_returns = labels.mean(axis=1)
                # Calculate cumulative market returns
                market_cumulative_returns = (1 + market_returns).cumprod() - 1
                logger.info(f"Market average returns calculated: {len(market_returns)} days")
            
            for strategy_name, result in results_dict.items():
                metrics = result['metrics']
                weights = result['weights']
                
                # Create main analysis charts
                fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                fig.suptitle(f'Strategy Performance Analysis: {strategy_name}', fontsize=16, fontweight='bold')
                
                # Convert date format - handle both datetime and string formats
                try:
                    if isinstance(metrics['cumulative_returns'].index[0], (pd.Timestamp, datetime)):
                        dates = [d for d in metrics['cumulative_returns'].index]
                    else:
                        dates = [datetime.strptime(str(d), '%Y%m%d') for d in metrics['cumulative_returns'].index]
                except:
                    # Fallback: try to convert to datetime directly
                    dates = pd.to_datetime(metrics['cumulative_returns'].index)
                
                # 1. Cumulative Returns with Market Benchmark
                axes[0, 0].plot(dates, metrics['cumulative_returns'].values, 'b-', linewidth=2, label='Strategy')
                
                # Add market benchmark if available
                if market_cumulative_returns is not None:
                    # Align market returns with strategy dates
                    market_aligned = market_cumulative_returns.reindex(metrics['cumulative_returns'].index)
                    if not market_aligned.empty:
                        # Handle date format for market data
                        try:
                            if isinstance(market_aligned.index[0], (pd.Timestamp, datetime)):
                                market_dates = [d for d in market_aligned.index]
                            else:
                                market_dates = [datetime.strptime(str(d), '%Y%m%d') for d in market_aligned.index]
                        except:
                            market_dates = pd.to_datetime(market_aligned.index)
                        
                        axes[0, 0].plot(market_dates, market_aligned.values, 'r--', linewidth=2, alpha=0.8, label='Market Average')
                        axes[0, 0].legend()
                
                axes[0, 0].set_title('Cumulative Returns vs Market')
                axes[0, 0].set_ylabel('Cumulative Returns')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                
                # 2. Drawdown Analysis
                peak = metrics['cumulative_returns'].expanding().max()
                drawdown = metrics['cumulative_returns'] - peak
                axes[0, 1].fill_between(dates, 0, drawdown.values, alpha=0.3, color='red')
                axes[0, 1].plot(dates, drawdown.values, 'r-', linewidth=1)
                axes[0, 1].set_title(f"Drawdown Analysis (Max DD: {metrics['max_drawdown']:.2%})")
                axes[0, 1].set_ylabel('Drawdown')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                
                # 3. Turnover Rate
                axes[0, 2].plot(dates, metrics['turnover_series'].values, 'g-', linewidth=1, alpha=0.7)
                axes[0, 2].axhline(y=metrics['turnover_series'].mean(), color='r', linestyle='--',
                                 label=f'Mean: {metrics["turnover_series"].mean():.2%}')
                axes[0, 2].set_title('Turnover Rate')
                axes[0, 2].set_ylabel('Turnover Rate')
                axes[0, 2].grid(True, alpha=0.3)
                axes[0, 2].legend()
                axes[0, 2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                
                # 4. Excess Cumulative Returns (超额累计收益曲线)
                if market_cumulative_returns is not None:
                    # 对齐市场累计收益和策略累计收益
                    market_aligned = market_cumulative_returns.reindex(metrics['cumulative_returns'].index)
                    excess_cumulative = metrics['cumulative_returns'] - market_aligned
                    # 日期处理
                    try:
                        if isinstance(metrics['cumulative_returns'].index[0], (pd.Timestamp, datetime)):
                            excess_dates = [d for d in metrics['cumulative_returns'].index]
                        else:
                            excess_dates = [datetime.strptime(str(d), '%Y%m%d') for d in metrics['cumulative_returns'].index]
                    except:
                        excess_dates = pd.to_datetime(metrics['cumulative_returns'].index)
                    axes[1, 0].plot(excess_dates, excess_cumulative.values, 'm-', linewidth=2, label='Excess Cumulative')
                    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                    axes[1, 0].set_title('Excess Cumulative Returns')
                    axes[1, 0].set_ylabel('Excess Cumulative Returns')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(0.5, 0.5, 'No Market Data', ha='center', va='center', fontsize=12)
                    axes[1, 0].set_title('Excess Cumulative Returns')
                    axes[1, 0].set_ylabel('Excess Cumulative Returns')
                    axes[1, 0].grid(True, alpha=0.3)
                
                # 5. Key Metrics Table
                axes[1, 1].axis('off')
                
                # Calculate market metrics if available
                market_total_return = None
                market_annualized_return = None
                excess_return = None
                excess_annualized_return = None
                
                if market_cumulative_returns is not None:
                    # Align market returns with strategy dates
                    market_aligned = market_cumulative_returns.reindex(metrics['cumulative_returns'].index)
                    if not market_aligned.empty:
                        market_total_return = market_aligned.iloc[-1]
                        market_annualized_return = (1 + market_total_return) ** (self.trading_days_per_year / len(market_aligned)) - 1
                        excess_return = metrics['total_return'] - market_total_return
                        excess_annualized_return = metrics['annualized_return'] - market_annualized_return
                
                # Build metrics text with market comparison
                if market_total_return is not None:
                    metrics_text = (
                        f"Total Return: {metrics['total_return']:.2%} (Market: {market_total_return:.2%})\n"
                        f"Excess Return: {excess_return:.2%}\n"
                        f"Annualized Return: {metrics['annualized_return']:.2%} (Market: {market_annualized_return:.2%})\n"
                        f"Excess Annualized: {excess_annualized_return:.2%}\n"
                        f"Annualized Volatility: {metrics['volatility']:.2%}\n"
                        f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n"
                        f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                        f"Calmar Ratio: {metrics['calmar_ratio']:.3f}\n"
                        f"Win Rate: {metrics['win_rate']:.2%}\n"
                        f"Profit/Loss Ratio: {metrics['profit_loss_ratio']:.2f}\n"
                        f"Avg Turnover: {metrics['turnover_series'].mean():.2%}\n"
                        f"Avg Holdings: {metrics['strategy_config']['avg_selected_stocks']:.1f}"
                    )
                else:
                    metrics_text = (
                        f"Total Return: {metrics['total_return']:.2%}\n"
                        f"Annualized Return: {metrics['annualized_return']:.2%}\n"
                        f"Annualized Volatility: {metrics['volatility']:.2%}\n"
                        f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n"
                        f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                        f"Calmar Ratio: {metrics['calmar_ratio']:.3f}\n"
                        f"Win Rate: {metrics['win_rate']:.2%}\n"
                        f"Profit/Loss Ratio: {metrics['profit_loss_ratio']:.2f}\n"
                        f"Avg Turnover: {metrics['turnover_series'].mean():.2%}\n"
                        f"Avg Holdings: {metrics['strategy_config']['avg_selected_stocks']:.1f}"
                    )
                
                axes[1, 1].text(0.1, 0.5, 'Key Metrics:\n\n' + metrics_text, 
                               fontsize=11, verticalalignment='center',
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                
                # 6. Daily Selection Statistics
                if 'daily_metrics' in metrics:
                    daily_df = metrics['daily_metrics']
                    if not daily_df.empty:
                        # Handle date format for daily metrics
                        try:
                            if isinstance(daily_df['date'].iloc[0], (pd.Timestamp, datetime)):
                                plot_dates = [d for d in daily_df['date']]
                            else:
                                plot_dates = [datetime.strptime(str(d), '%Y%m%d') for d in daily_df['date']]
                        except:
                            plot_dates = pd.to_datetime(daily_df['date'])
                        
                        axes[1, 2].plot(plot_dates, daily_df['selected_count'], 'orange', linewidth=2, label='Selected Count')
                        axes[1, 2].plot(plot_dates, daily_df['valid_labels_count'], 'purple', linewidth=2, alpha=0.7, label='Valid Labels Count')
                        axes[1, 2].set_title('Daily Selection Statistics')
                        axes[1, 2].set_ylabel('Count')
                        axes[1, 2].grid(True, alpha=0.3)
                        axes[1, 2].legend()
                        axes[1, 2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                
                # Set x-axis rotation for all plots
                for ax in axes.flat:
                    if hasattr(ax, 'xaxis'):
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                plt.tight_layout()
                
                # Save charts
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(os.path.join(save_path, f'{strategy_name}_enhanced_analysis.png'), 
                              dpi=300, bbox_inches='tight')
                    logger.info(f"Chart saved: {save_path}/{strategy_name}_enhanced_analysis.png")
                
                plt.show()
            
            # Create decile and quintile analysis only once (since it's independent of strategy selection)
            if pred_returns is not None and labels is not None and not group_analysis_created:
                logger.info("Creating group analysis (independent of strategy selection)")
                self._create_group_analysis_plots("Model Prediction", pred_returns, labels, save_path)
                group_analysis_created = True
                
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")

    def _create_group_analysis_plots(self, strategy_name: str, pred_returns: pd.DataFrame, 
                                   labels: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create decile and quintile group analysis plots
        
        Args:
            strategy_name: Strategy name
            pred_returns: Predicted returns matrix
            labels: True labels matrix
            save_path: Save path (optional)
        """
        try:
            logger.info(f"Creating group analysis for {strategy_name}")
            
            # Data alignment
            labels_align = labels.reindex(index=pred_returns.index, columns=pred_returns.columns)
            
            # Calculate group returns
            decile_returns = self._calculate_group_returns(pred_returns, labels_align, n_groups=10)
            quintile_returns = self._calculate_group_returns(pred_returns, labels_align, n_groups=5)
            
            # Create group analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle(f'Model Prediction Group Analysis', fontsize=16, fontweight='bold')
            
            # Convert dates
            dates = [datetime.strptime(str(d), '%Y%m%d') for d in decile_returns.index]
            
            # 1. Decile cumulative returns
            colors_decile = plt.cm.RdYlGn(np.linspace(0, 1, 10))
            for i in range(10):
                group_name = f'D{i+1}'
                cumulative = (1 + decile_returns.iloc[:, i]).cumprod() - 1
                axes[0, 0].plot(dates, cumulative.values, 
                              color=colors_decile[i], linewidth=2, label=group_name)
            
            axes[0, 0].set_title('Decile Group Cumulative Returns', fontsize=14)
            axes[0, 0].set_ylabel('Cumulative Returns')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # 2. Quintile cumulative returns
            colors_quintile = plt.cm.RdYlGn(np.linspace(0, 1, 5))
            for i in range(5):
                group_name = f'Q{i+1}'
                cumulative = (1 + quintile_returns.iloc[:, i]).cumprod() - 1
                axes[0, 1].plot(dates, cumulative.values, 
                              color=colors_quintile[i], linewidth=3, label=group_name)
            
            axes[0, 1].set_title('Quintile Group Cumulative Returns', fontsize=14)
            axes[0, 1].set_ylabel('Cumulative Returns')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # 3. Decile annualized returns bar chart
            decile_annual_returns = []
            for i in range(10):
                annual_ret = (1 + decile_returns.iloc[:, i]).prod() ** (252 / len(decile_returns)) - 1
                decile_annual_returns.append(annual_ret)
            
            bars = axes[1, 0].bar(range(1, 11), decile_annual_returns, 
                                color=colors_decile, alpha=0.8)
            axes[1, 0].set_title('Decile Annualized Returns', fontsize=14)
            axes[1, 0].set_xlabel('Decile Group')
            axes[1, 0].set_ylabel('Annualized Return')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xticks(range(1, 11))
            axes[1, 0].set_xticklabels([f'D{i}' for i in range(1, 11)])
            
            # Add value labels on bars
            for bar, value in zip(bars, decile_annual_returns):
                height = bar.get_height()
                axes[1, 0].annotate(f'{value:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3), textcoords="offset points",
                                  ha='center', va='bottom', fontsize=9)
            
            # 4. Quintile annualized returns bar chart
            quintile_annual_returns = []
            for i in range(5):
                annual_ret = (1 + quintile_returns.iloc[:, i]).prod() ** (252 / len(quintile_returns)) - 1
                quintile_annual_returns.append(annual_ret)
            
            bars = axes[1, 1].bar(range(1, 6), quintile_annual_returns, 
                                color=colors_quintile, alpha=0.8)
            axes[1, 1].set_title('Quintile Annualized Returns', fontsize=14)
            axes[1, 1].set_xlabel('Quintile Group')
            axes[1, 1].set_ylabel('Annualized Return')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xticks(range(1, 6))
            axes[1, 1].set_xticklabels([f'Q{i}' for i in range(1, 6)])
            
            # Add value labels on bars
            for bar, value in zip(bars, quintile_annual_returns):
                height = bar.get_height()
                axes[1, 1].annotate(f'{value:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3), textcoords="offset points",
                                  ha='center', va='bottom', fontsize=10)
            
            # Set x-axis rotation for all plots
            for ax in axes.flat:
                if hasattr(ax, 'xaxis'):
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save group analysis chart
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'model_prediction_group_analysis.png'), 
                          dpi=300, bbox_inches='tight')
                logger.info(f"Group analysis chart saved: {save_path}/model_prediction_group_analysis.png")
            
            plt.show()
            
            # Print group analysis summary
            print(f"\n=== Model Prediction Group Analysis Summary ===")
            print("(This analysis shows the model's ability to rank stocks by future returns)")
            print("\nDecile Analysis (All stocks divided into 10 equal groups by prediction score):")
            for i, ret in enumerate(decile_annual_returns):
                print(f"  D{i+1} (Top {10*(i+1):2.0f}%): {ret:8.2%}")
            
            print("\nQuintile Analysis (All stocks divided into 5 equal groups by prediction score):")
            for i, ret in enumerate(quintile_annual_returns):
                print(f"  Q{i+1} (Top {20*(i+1):2.0f}%): {ret:8.2%}")
            
            # Calculate spread metrics
            decile_spread = decile_annual_returns[0] - decile_annual_returns[-1]
            quintile_spread = quintile_annual_returns[0] - quintile_annual_returns[-1]
            
            print(f"\nModel Effectiveness Analysis:")
            print(f"  Decile Spread (D1 - D10): {decile_spread:8.2%}")
            print(f"  Quintile Spread (Q1 - Q5): {quintile_spread:8.2%}")
            
            # Model effectiveness assessment
            if decile_spread > 0.1:  # 10%+ spread
                effectiveness = "Excellent"
            elif decile_spread > 0.05:  # 5%+ spread
                effectiveness = "Good"
            elif decile_spread > 0.02:  # 2%+ spread
                effectiveness = "Fair"
            else:
                effectiveness = "Poor"
            
            print(f"  Model Ranking Ability: {effectiveness}")
            print(f"  (A good model should show higher returns for higher-ranked groups)")
            
        except Exception as e:
            logger.error(f"Group analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_group_returns(self, pred_returns: pd.DataFrame, labels: pd.DataFrame, 
                               n_groups: int = 10) -> pd.DataFrame:
        """
        Calculate group returns for decile/quintile analysis (Fully Vectorized Version)
        
        Args:
            pred_returns: Predicted returns matrix
            labels: True labels matrix
            n_groups: Number of groups (10 for decile, 5 for quintile)
            
        Returns:
            pd.DataFrame: Group returns matrix [time, group]
        """
        try:
            logger.info(f"Calculating {n_groups}-group returns using fully vectorized method")
            
            # Stack data for vectorized operations
            pred_stacked = pred_returns.stack()  # Multi-index: (date, stock)
            labels_stacked = labels.stack()      # Multi-index: (date, stock)
            
            # Create aligned DataFrame
            aligned_data = pd.DataFrame({
                'pred': pred_stacked,
                'label': labels_stacked
            }).dropna()
            
            if aligned_data.empty:
                logger.warning("No valid data after alignment")
                return pd.DataFrame(
                    index=pred_returns.index, 
                    columns=[f'Group_{i+1}' for i in range(n_groups)],
                    dtype=float
                ).fillna(0)
            
            logger.info(f"Processing {len(aligned_data)} valid data points across {len(pred_returns.index)} dates")
            
            # Reset index to make date a column
            aligned_data = aligned_data.reset_index()
            aligned_data.columns = ['date', 'stock', 'pred', 'label']
            
            # Vectorized ranking within each date
            logger.info("Computing rankings...")
            aligned_data['rank'] = aligned_data.groupby('date')['pred'].rank(method='first', ascending=False)
            
            # Calculate group assignments vectorized
            logger.info("Assigning groups...")
            def assign_group_vectorized(group_df):
                n_stocks = len(group_df)
                if n_stocks < n_groups:
                    # Not enough stocks, assign all to group 1
                    return pd.Series([1] * n_stocks, index=group_df.index)
                
                # Calculate group boundaries
                group_size = n_stocks / n_groups
                group_assignments = ((group_df['rank'] - 1) // group_size).astype(int) + 1
                # Ensure no group exceeds n_groups
                group_assignments = np.minimum(group_assignments, n_groups)
                return group_assignments
            
            # Apply group assignment
            aligned_data['group'] = aligned_data.groupby('date').apply(
                lambda x: assign_group_vectorized(x)
            ).values
            
            # Calculate group returns using pivot_table (fully vectorized)
            logger.info("Calculating group returns...")
            group_results = aligned_data.pivot_table(
                index='date',
                columns='group', 
                values='label',
                aggfunc='mean'
            )
            
            # Ensure all groups are present and properly named
            all_groups = list(range(1, n_groups + 1))
            missing_groups = set(all_groups) - set(group_results.columns)
            
            # Add missing groups with zero returns
            for group in missing_groups:
                group_results[group] = 0.0
            
            # Reorder columns and rename
            group_results = group_results[all_groups]
            group_results.columns = [f'Group_{i}' for i in all_groups]
            
            # Reindex to include all dates from original data
            group_results = group_results.reindex(pred_returns.index, fill_value=0.0)
            
            logger.info(f"Group returns calculation completed. Shape: {group_results.shape}")
            
            # Validation
            non_zero_days = (group_results != 0).any(axis=1).sum()
            total_days = len(group_results)
            logger.info(f"Non-zero return days: {non_zero_days}/{total_days} ({non_zero_days/total_days:.1%})")
            
            # Performance statistics
            for i in range(n_groups):
                col = f'Group_{i+1}'
                mean_ret = group_results[col].mean()
                std_ret = group_results[col].std()
                logger.info(f"{col}: Mean={mean_ret:.4f}, Std={std_ret:.4f}")
            
            return group_results
            
        except Exception as e:
            logger.error(f"Vectorized group returns calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(
                index=pred_returns.index, 
                columns=[f'Group_{i+1}' for i in range(n_groups)],
                dtype=float
            ).fillna(0)


# 兼容性函数 - 保持原有接口
def backtest_strategy(pred_returns, labels, market_cap=None):
    """
    兼容性函数，保持原有接口
    """
    engine = BacktestEngine()
    weights, results = engine.backtest_strategy(pred_returns, labels, market_cap)
    
    # 转换为原有格式
    return weights, {
        'cumulative_returns': results['cumulative_returns'],
        'max_drawdown': results['max_drawdown'],
        'annualized_return': results['annualized_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'turnover_series': results['turnover_series']
    }


def plot_model_metrics_and_save(pred_df_list, save_path):
    """
    兼容性函数 - 使用新的绘图引擎
    """
    engine = BacktestEngine()
    
    # 转换数据格式
    results_dict = {}
    for model_name, (weights, metrics) in pred_df_list.items():
        results_dict[model_name] = {
            'weights': weights,
            'metrics': metrics
        }
    
    engine.create_enhanced_plots(results_dict, save_path)


def plot_model_metrics(pred_df_list):
    """
    兼容性函数 - 使用新的绘图引擎
    """
    plot_model_metrics_and_save(pred_df_list, None)


def analyze_groups(pred_returns, labels, save_path=None, create_plots=True):
    """
    模型预测能力分组分析函数
    
    通过将所有股票按预测收益率分成十分组和五分组，分析模型的排序能力。
    这个分析独立于具体的选股策略（如选择前5%还是前10%），
    主要用于评估模型是否能有效区分好股票和坏股票。
    
    Args:
        pred_returns (pd.DataFrame): 预测收益率矩阵
        labels (pd.DataFrame): 真实标签矩阵
        save_path (str, optional): 保存路径
        create_plots (bool, optional): 是否创建图表，默认True
        
    Returns:
        dict: 包含十分组和五分组分析结果
    """
    engine = BacktestEngine()
    
    # 数据对齐
    labels_align = labels.reindex(index=pred_returns.index, columns=pred_returns.columns)
    
    # 计算分组收益
    decile_returns = engine._calculate_group_returns(pred_returns, labels_align, n_groups=10)
    quintile_returns = engine._calculate_group_returns(pred_returns, labels_align, n_groups=5)
    
    # 创建分组分析图表（可选）
    if create_plots:
        engine._create_group_analysis_plots("Model Prediction", pred_returns, labels_align, save_path)
    
    # 返回分析结果
    return {
        'decile_returns': decile_returns,
        'quintile_returns': quintile_returns,
        'decile_cumulative': (1 + decile_returns).cumprod() - 1,
        'quintile_cumulative': (1 + quintile_returns).cumprod() - 1
    }


if __name__ == '__main__':
    """
    使用示例和测试代码
    """
    try:
        # 原有测试代码保持不变
        pred_path = r'D:\chenxing\Finforecast\exp\rolling_pred\AttGRU_112_202401_202504.csv'
        save_path = r'D:\chenxing\Finforecast\exp\rolling_pred\AttGRU_112_202401_202504'
        labels_list = [f'D:/chenxing/Finforecast/factor_warehouse/factor_aligned/r_label/{i}/label.parquet' for i in range(202401, 202505) if i % 100 <= 12 and i % 100 != 0]
        labels_df = pd.concat([pd.read_parquet(i) for i in labels_list], axis=0)
        pred_df = pd.read_csv(pred_path, index_col=0).astype(np.float32)
        print(pred_df.shape)
        print(labels_df.shape)
        
        # 转换数据类型
        labels_df.index = labels_df.index.astype(np.int32)
        pred_df.index = pred_df.index.astype(np.int32)
        labels_df.columns = labels_df.columns.astype(np.int32)
        pred_df.columns = pred_df.columns.astype(np.int32)
        
        print("=" * 80)
        print("Enhanced Backtest Example")
        print("=" * 80)
        
        # Create backtest engine
        engine = BacktestEngine()
        
        # Single strategy backtest
        print("\n[Single Strategy Backtest]")
        weights, results = engine.backtest_strategy(pred_df, labels_df, top_pct=0.1)
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        # Multiple strategy comparison example
        print("\n[Multiple Strategy Comparison]")
        strategies = [
            # {'name': 'Top 5% Strategy', 'top_pct': 0.05, 'weight_method': 'equal'},
            {'name': 'Top 10% Strategy', 'top_pct': 0.1, 'weight_method': 'equal'},
            # {'name': 'Top 15% Strategy', 'top_pct': 0.15, 'weight_method': 'equal'},
        ]
        
        comparison_results = engine.compare_strategies(pred_df, labels_df, strategies)
        
        # Print comparison results
        for name, result in comparison_results.items():
            metrics = result['metrics']
            print(f"{name}: Annualized Return {metrics['annualized_return']:.2%}, "
                  f"Sharpe {metrics['sharpe_ratio']:.3f}, "
                  f"Drawdown {metrics['max_drawdown']:.2%}")
        
        # Create charts with group analysis
        engine.create_enhanced_plots(comparison_results, pred_returns=pred_df, labels=labels_df, save_path=save_path)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
    