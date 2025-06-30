# Finforecast - 量化金融Alpha预测系统

[![Python](https://img.shields.io/badge/Python-3.10.17-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu118-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 项目概述

Finforecast是一个专业的量化金融Alpha预测机器学习框架，专门用于股票因子建模和收益率预测。该系统集成了深度时序模型和传统机器学习模型，支持多种因子数据源，提供完整的数据处理、模型训练、评估和预测流水线。

### 🎯 核心特性

- **多模型架构**：时序模型（GRU、BiGRU、AttGRU、TimeMixer）+ 树模型（LightGBM、XGBoost、CatBoost）
- **多因子支持**：支持Ding128、CY312、DrJin129、CCB、分钟级因子等多种因子库
- **滚动训练**：支持时间窗口滚动训练，防止数据泄露
- **专业评估**：IC分析、分档回测、风险指标计算
- **高性能**：GPU加速训练，批量预测优化
- **模块化设计**：数据处理、模型训练、评估预测完全解耦

### 💾 多因子库支持

系统支持多种不同特性的因子库，满足不同研究需求：

#### 📊 CY312 因子库
- **数据格式**: Pickle存储，每个因子[B,T]格式
- **特色功能**: 
  - 支持标签对齐(lbl_align)和市值对齐(mkt_align)两种模式
  - 24线程并行数据处理，大幅提升加载速度
  - 完整的滚动训练支持
- **适用场景**: 大规模因子研究，需要高性能数据处理

#### 🔄 DrJin129 因子库  
- **数据格式**: NPY存储，[C,B,T]格式
- **特色功能**:
  - **双时间窗口设计**：
    - 半日因子(halfday)：利用交易时间之前的数据计算
    - 全日因子(allday)：利用全天数据计算
  - **防数据泄露**：通过时间分割确保预测的时间一致性
  - 支持标签和市值两种对齐方式
- **适用场景**: 需要严格防止数据泄露的高频交易研究

#### 🏦 Ding128 因子库  
- **数据格式**: 多类型混合存储
- **特色功能**:
  - **灵活对齐方式**：
    - ALB模式：与标签对齐
    - AMC模式：与市值对齐
- **适用场景**: 需要多维度因子分析的精细化研究

#### 🏛️ CCB 因子库
- **因子数量**: 可配置
- **数据格式**: 统一Parquet格式
- **特色功能**:
  - 最新的数据架构设计
  - 优化的内存使用和加载性能
  - 支持大规模数据处理
- **适用场景**: 新一代因子研究平台

#### ⏱️ 分钟级因子库
- **数据频率**: 分钟级高频数据
- **数据格式**: [B,T]格式的NPY文件
- **特色功能**:
  - 高频数据处理优化
  - 支持因子重命名和映射表管理
  - 专门的截面数据加载器
- **适用场景**: 高频量化策略，日内交易研究

## 🏗️ 系统架构

```
Finforecast/
├── 📁 get_data/          # 数据获取模块
│   ├── CCB/              # CCB因子数据加载器
│   ├── CY312/            # CY312因子数据加载器  
│   ├── DrJin129/         # DrJin129因子数据加载器
│   ├── Ding128/          # Ding128因子数据加载器
│   ├── data_align_and_save/  # 数据对齐和保存工具
│   └── minute_factors/   # 分钟级因子处理
├── 📁 model/             # 模型库
│   ├── GRU_model.py      # GRU相关模型
│   ├── GRU_attention.py  # 注意力GRU模型
│   ├── TimeMixer.py      # TimeMixer时序模型
│   ├── Encoder.py        # 编码器组件
│   ├── losses.py         # 损失函数库
│   └── layers/           # 模型层组件
├── 📁 train/             # 训练模块
│   ├── GRU_cross_time_train.py  # 时序模型训练
│   └── GBDT_trainer.py   # 树模型训练
├── 📁 metrics/           # 评估模块  
│   ├── calculate_ic.py   # IC计算
│   ├── backtest.py       # 回测分析
│   ├── train_plot.py     # 训练可视化
│   └── models_pred.py    # 模型预测
├── 📁 utils/             # 工具模块
│   ├── tools.py          # 通用工具
│   ├── fillna.py         # 缺失值处理  
│   ├── data_analysis.py  # 数据分析
│   └── metrics.py        # 评估指标
├── 📁 exp/               # 实验结果（.gitignore）
├── 📁 factor_warehouse/  # 因子仓库（.gitignore）
├── config.py             # 全局配置文件
├── GRU_GBDT_rollingtrain.py      # 混合模型滚动训练
├── fintest_savepred.py   # 最终预测保存
└── requirements.txt      # 依赖管理
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.10.17
- **PyTorch**: 2.6.0+cu118 (GPU加速推荐)
- **操作系统**: Linux/Windows/macOS
- **内存**: 16GB+ 推荐
- **GPU**: CUDA 11.8+ (可选，大幅提升训练速度)

### 安装步骤

1. **克隆仓库**
```bash
git clone git@github.com:keyboard1st/Finforecast.git
cd Finforecast
```

2. **创建虚拟环境**
```bash
conda create -n finforecast python=3.10.17
conda activate finforecast
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **验证安装**
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 数据准备

项目支持多种因子数据格式，需要先将原始数据转换为统一的Parquet格式：

```bash
# 针对不同因子库运行相应的数据处理脚本
python get_data/data_align_and_save/save_CY312.py      # CY312因子
python get_data/data_align_and_save/save_DrJin129_rolling.py  # DrJin129因子  
python get_data/data_align_and_save/save_minute_features.py   # 分钟级因子
```

## 🔧 使用指南

### 1. 配置实验参数

在 `config.py` 中修改实验配置：

```python
# 主要参数示例
model_type = 'AttGRU'                    # 模型类型
time_period = '2019-2025'                # 训练时间段
device = 'cuda:0'                        # 计算设备
window_size = 30                         # 时间窗口大小
hidden_dim = 128                         # 隐藏层维度
learning_rate = 0.0001                   # 学习率
```

### 2. 模型训练

#### 批量训练（推荐）
```bash
# 并行训练多个滚动训练任务
bash models_train_demo.sh
```

#### 单独训练
```bash
# 仅训练时序模型
python GRU_train_ICloss_new.py

# 仅训练树模型
python GBDT_rollingtrain.py  

# 混合模型训练
python GRU_GBDT_rollingtrain.py
```

### 3. 模型预测

```bash
# 批量预测和结果保存
bash models_pred_demo.sh

# 单独预测
python fintest_savepred.py
```

### 4. 结果分析

训练和预测结果保存在 `exp/{task_name}/` 目录下：

```
exp/CCB_2019_2025_AttGRU/
├── models/           # 模型权重文件
├── plots/            # 训练过程图表
├── pred_csv/         # 预测结果CSV
├── exp.xlsx          # 实验指标汇总
└── *.log            # 训练日志
```

## 📈 评估指标

系统提供专业的量化评估指标：

- **IC (Information Coefficient)**: 预测值与真实收益的相关性
- **年化收益率**: 策略年化收益
- **最大回撤**: 策略最大回撤
- **夏普比率**: 风险调整后收益
- **换手率**: 策略交易频率
- **分档回测**: 十档收益分析
- **累计净值曲线**: 策略表现可视化


## 📧 联系方式

- **作者**: keyboard1st
- **邮箱**: 157776579@qq.com
- **项目地址**: https://github.com/keyboard1st/Finforecast

   