import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Dict, List, Union


class TrainingMetrics:
    def __init__(self, save_dir: str = "./plots"):
        """
        训练指标可视化类
        :param save_dir: 图表保存目录
        """
        self.metrics: Dict[str, List] = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'val_ic_mean': [],
            'test_ic_mean': [],
            'lr': []
        }
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 绘图样式配置
        self.style_config = {
            'loss': {
                'colors': ['b-', 'r--', 'g--'],
                'labels': ['Train Loss', 'Val Loss', 'Test Loss'],
                'ylim_buffer': 0.1
            },
            'ic': {
                'markers': {
                    'val': ('r-', 'o', 'g'),
                    'test': ('b-', 'o', 'b')
                },
                'ylim_range': (-1.0, 1.0)
            },
            'lr': {
                'color': 'b-',
                'ylim_buffer': 0.1
            }
        }

    def add_metrics(self, epoch: int, **kwargs):
        """
        添加指标数据
        :param epoch: 当前epoch数
        :param kwargs: 需要记录的指标键值对
        """
        self.metrics['epochs'].append(epoch)

        for metric_name, value in kwargs.items():
            if metric_name in self.metrics:
                if isinstance(value, (list, tuple)):
                    self.metrics[metric_name].extend(value)
                else:
                    self.metrics[metric_name].append(value)
            else:
                raise KeyError(f"Invalid metric name: {metric_name}")

    def plot_all(self, prefix: str = ""):
        """生成全部图表"""
        self.plot_loss(prefix)
        self.plot_ic(prefix)
        self.plot_lr(prefix)

    def plot_loss(self, prefix: str = ""):
        """绘制损失曲线"""
        plt.figure(figsize=(8, 6))

        # 绘制三条损失曲线
        for style, label, key in zip(
                self.style_config['loss']['colors'],
                self.style_config['loss']['labels'],
                ['train_loss', 'val_loss', 'test_loss']
        ):
            plt.plot(self.metrics['epochs'], self.metrics[key], style,
                     lw=1.5, label=label)

        # 动态设置Y轴范围
        loss_values = []
        for k in ['train_loss', 'val_loss', 'test_loss']:
            loss_values.extend(self.metrics[k])

        y_min = min(loss_values) * (1 - self.style_config['loss']['ylim_buffer'])
        y_max = max(loss_values) * (1 + self.style_config['loss']['ylim_buffer'])
        plt.ylim(y_min, y_max)

        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Training Loss Curves')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        self._save_plot('loss', prefix)

    def plot_ic(self, prefix: str = ""):
        """绘制IC曲线"""
        plt.figure(figsize=(8, 6))

        # 绘制双IC曲线
        for metric in ['val_ic_mean', 'test_ic_mean']:
            config_key = metric.split('_')[0]  # 提取val/test
            style, marker, edgecolor = self.style_config['ic']['markers'][config_key]

            plt.plot(self.metrics['epochs'], self.metrics[metric],
                     style, lw=1.5, marker=marker, markersize=4,
                     markerfacecolor='white', markeredgecolor=edgecolor,
                     label=f"{config_key.title()} IC Mean")

        # 设置Y轴范围
        plt.ylim(
            max(self.style_config['ic']['ylim_range'][0],
                min(self.metrics['val_ic_mean'] + self.metrics['test_ic_mean']) - 0.1),
            min(self.style_config['ic']['ylim_range'][1],
                max(self.metrics['val_ic_mean'] + self.metrics['test_ic_mean']) + 0.1)
        )

        plt.xlabel('Epoch')
        plt.ylabel('Information Coefficient')
        plt.title('IC Performance Curves')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        self._save_plot('ic', prefix)

    def plot_lr(self, prefix: str = ""):
        """绘制学习率曲线"""
        plt.figure(figsize=(8, 6))

        plt.plot(self.metrics['epochs'], self.metrics['lr'],
                 self.style_config['lr']['color'], lw=1.5)

        # 动态设置Y轴范围
        lr_values = self.metrics['lr']
        buffer = self.style_config['lr']['ylim_buffer']
        plt.ylim(min(lr_values) * (1 - buffer), max(lr_values) * (1 + buffer))

        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, linestyle='--', alpha=0.6)

        self._save_plot('lr', prefix)

    def _save_plot(self, plot_type: str, prefix: str):
        """统一保存处理"""
        filename = f"{prefix}_{plot_type}_curve.png" if prefix else f"{plot_type}_curve.png"
        save_path = os.path.join(self.save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def to_dataframe(self):
        # 确保所有列表长度相同
        max_len = max(len(v) for v in self.metrics.values())
        for key in self.metrics:
            if len(self.metrics[key]) < max_len:
                self.metrics[key].extend([None] * (max_len - len(self.metrics[key])))

        return pd.DataFrame(self.metrics)


def plot_training_metrics(metrics, save_path):
    plt.figure(figsize=(12, 6))

    # 合并训练和测试损失曲线 (左图)
    plt.subplot(1, 3, 1)
    plt.plot(metrics['epochs'], metrics['train_loss'], 'b-', lw=1.5, label='Train Loss')
    plt.plot(metrics['epochs'], metrics['val_loss'], 'r--', lw=1.5, label='Val Loss')
    plt.plot(metrics['epochs'], metrics['test_loss'], 'g--', lw=1.5, label='Test Loss')

    # 设置坐标轴范围
    min_loss = min(min(metrics['train_loss']), min(metrics['val_loss']), min(metrics['test_loss']))
    max_loss = max(max(metrics['train_loss']), max(metrics['val_loss']), max(metrics['test_loss']))
    plt.ylim(min_loss*0.9, max_loss*1.1)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training & Val & Test Loss Curves', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)

    # IC均值曲线 (右图)
    plt.subplot(1, 3, 2)
    plt.plot(metrics['epochs'], metrics['val_ic_mean'], 'r-', lw=1.5, marker='o', markersize=4, markerfacecolor='white', markeredgecolor='g', label='Val IC Mean')
    plt.plot(metrics['epochs'], metrics['test_ic_mean'], 'b-', lw=1.5, marker='o', markersize=4, markerfacecolor='white', markeredgecolor='b', label='Test IC Mean')


    # 智能设置IC范围
    ic_min = min(min(metrics['val_ic_mean']), min(metrics['test_ic_mean']))
    ic_max = max(max(metrics['val_ic_mean']), max(metrics['test_ic_mean']))
    plt.ylim(max(-1.0, ic_min - 0.1), min(1.0, ic_max + 0.1))

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Information Coefficient', fontsize=12)
    plt.title('IC Mean Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True)

    # LR曲线
    plt.subplot(1, 3, 3)
    plt.plot(metrics['lr'], 'b-', lw=1.5)
    lr_min = min(metrics['lr'])
    lr_max = max(metrics['lr'])
    plt.ylim(lr_min*0.9, lr_max*1.1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 调整布局并保存
    plt.tight_layout(pad=3.0)
    plot_path = os.path.join(save_path, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存原始数据
    pd.DataFrame({
        'epoch': metrics['epochs'],
        'train_loss': metrics['train_loss'],
        'val_loss': metrics['val_loss'],
        'test_loss': metrics['test_loss'],
        'val_ic_mean': metrics['val_ic_mean'],
        'test_ic_mean': metrics['test_ic_mean']
    }).to_csv(os.path.join(save_path, 'training_metrics.csv'), index=False)

def plot_training_metrics_withoutlr(metrics, save_path):
    plt.figure(figsize=(12, 6))

    # 合并训练和测试损失曲线 (左图)
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epochs'], metrics['train_loss'], 'b-', lw=1.5, label='Train Loss')
    plt.plot(metrics['epochs'], metrics['val_loss'], 'r--', lw=1.5, label='Val Loss')
    plt.plot(metrics['epochs'], metrics['test_loss'], 'g--', lw=1.5, label='Test Loss')

    # 设置坐标轴范围
    min_loss = min(min(metrics['train_loss']), min(metrics['val_loss']), min(metrics['test_loss']))
    max_loss = max(max(metrics['train_loss']), max(metrics['val_loss']), max(metrics['test_loss']))
    plt.ylim(min_loss*0.9, max_loss*1.1)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training & Val & Test Loss Curves', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)

    # IC均值曲线 (右图)
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epochs'], metrics['val_ic_mean'], 'r-', lw=1.5, marker='o', markersize=4, markerfacecolor='white', markeredgecolor='g', label='Val IC Mean')
    plt.plot(metrics['epochs'], metrics['test_ic_mean'], 'b-', lw=1.5, marker='o', markersize=4, markerfacecolor='white', markeredgecolor='b', label='Test IC Mean')


    # 智能设置IC范围
    ic_min = min(min(metrics['val_ic_mean']), min(metrics['test_ic_mean']))
    ic_max = max(max(metrics['val_ic_mean']), max(metrics['test_ic_mean']))
    plt.ylim(max(-1.0, ic_min - 0.1), min(1.0, ic_max + 0.1))

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Information Coefficient', fontsize=12)
    plt.title('IC Mean Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True)

    # 调整布局并保存
    plt.tight_layout(pad=3.0)
    plot_path = os.path.join(save_path, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存原始数据
    pd.DataFrame({
        'epoch': metrics['epochs'],
        'train_loss': metrics['train_loss'],
        'val_loss': metrics['val_loss'],
        'test_loss': metrics['test_loss'],
        'val_ic_mean': metrics['val_ic_mean'],
        'test_ic_mean': metrics['test_ic_mean']
    }).to_csv(os.path.join(save_path, 'training_metrics.csv'), index=False)


def plot_loss_curves(metrics, save_path):
    """绘制损失曲线并单独保存"""
    plt.figure(figsize=(8, 6))

    # 绘制三条损失曲线
    lines = [
        ('b-', 'Train Loss', metrics['train_loss']),
        ('r--', 'Val Loss', metrics['val_loss']),
        ('g--', 'Test Loss', metrics['test_loss'])
    ]

    for style, label, data in lines:
        plt.plot(metrics['epochs'], data, style, lw=1.5, label=label)

    # 动态设置Y轴范围
    loss_values = metrics['train_loss'] + metrics['val_loss'] + metrics['test_loss']
    y_min = min(loss_values) * 0.9
    y_max = max(loss_values) * 1.1

    plt.ylim(y_min, y_max)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Metrics - Loss Curves', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)

    # 保存并清理
    plot_path = os.path.join(save_path, 'loss_curves.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ic_curves(metrics, save_path):
    """绘制IC均值曲线并单独保存"""
    plt.figure(figsize=(8, 6))

    # 绘制双IC曲线
    plt.plot(metrics['epochs'], metrics['val_ic_mean'],
             'r-', lw=1.5, marker='o', markersize=4,
             markerfacecolor='white', markeredgecolor='g',
             label='Val IC Mean')

    plt.plot(metrics['epochs'], metrics['test_ic_mean'],
             'b-', lw=1.5, marker='o', markersize=4,
             markerfacecolor='white', markeredgecolor='b',
             label='Test IC Mean')

    # 智能Y轴范围
    ic_values = metrics['val_ic_mean'] + metrics['test_ic_mean']
    y_min = max(-1.0, (min(ic_values) - 0.1))
    y_max = min(1.0, (max(ic_values) + 0.1))

    plt.ylim(y_min, y_max)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Information Coefficient', fontsize=12)
    plt.title('Validation IC Performance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True)

    # 保存并清理
    plot_path = os.path.join(save_path, 'ic_curves.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_lr_curve(metrics, save_path):
    """绘制学习率曲线并单独保存"""
    plt.figure(figsize=(8, 6))

    # 绘制学习率曲线
    plt.plot(metrics['epochs'], metrics['lr'],
             'b-', lw=1.5, label='Learning Rate')

    # 动态设置Y轴范围
    lr_values = metrics['lr']
    y_min = min(lr_values) * 0.9
    y_max = max(lr_values) * 1.1

    plt.ylim(y_min, y_max)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存并清理
    plot_path = os.path.join(save_path, 'lr_curve.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()