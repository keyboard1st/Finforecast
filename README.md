# Finforecast

## 项目概述

Finforecast是一个用于金融alpha预测的机器学习模型库，包含了多种时序学习模型，如GRU、BiGRU、AttGRU、TimeMixer等，以及GBDT、XGBoost、CatBoost等截面模型。项目还提供了数据加载、模型训练、模型评估、因子验证、滚动预测等功能。

## 核心功能

### 📊 数据处理 (get_data)
- **多源数据支持**：兼容不同格式的金融数据（Parquet、Pickle、NPY等）
- **数据对齐**：自动处理不同数据源的时间对齐和股票对齐
- **分层数据加载**：
  - 时序数据：支持变长序列，自动padding和masking
  - 截面数据：支持缺失值处理和标准化
  - 分钟级因子：高频数据的预处理和特征工程
- **数据分割**：按年份自动分割训练、验证、测试集
- **内存优化**：支持大规模数据的分块加载和处理

### 🤖 模型训练 (train)
- **混合模型架构**：时序模型+截面模型的联合训练
- **滚动训练**：支持walk-forward validation，避免未来信息泄露
- **损失函数优化**：
  - IC损失：直接优化Information Coefficient
  - Rank损失：优化预测值的排序性能
  - MSE损失：传统的均方误差损失
- **正则化技术**：Dropout、L1/L2正则、Early Stopping
- **跨平台支持**：Linux和Windows环境的兼容性

### 📈 评估系统 (metrics)
- **IC分析**：全面的Information Coefficient统计分析
- **分档回测**：
  - 10档分层回测，支持自定义分档数
  - 行业中性化处理
  - 多头/多空/市场中性策略
- **风险管理**：
  - 最大回撤控制
  - 夏普比率和信息比率监控
  - VaR风险价值计算
- **交易成本**：真实的交易费用和冲击成本建模
- **可视化**：自动生成回测报告和图表

### 🧠 模型库 (model)
- **深度学习模型**：
  - GRU/LSTM/Transformer系列
  - 注意力机制和残差连接
  - 时间序列专用编码器
- **传统机器学习**：
  - 梯度提升树（LightGBM、XGBoost、CatBoost）
  - 支持GPU加速和分布式训练
- **模型融合**：多种融合策略提升预测精度
- **自动调参**：集成Optuna等超参数优化框架

### 🛠️ 工具集 (utils)
- **特征工程**：时间特征、技术指标、统计特征
- **数据质量**：异常值检测、缺失值处理、数据验证
- **时间处理**：交易日历、时间窗口、滚动计算
- **并行计算**：多进程和多线程优化

## 项目结构

项目包含以下目录和文件：

### 核心配置和主文件
- **config.py**：总配置文件，包含实验参数、模型参数、数据路径等，更换所有实验参数只需要修改这个文件。
- **fintest_savepred.py**：最终模型推理和保存预测结果的主文件。
- **rolling_pred_concat.py**：滚动预测结果连接和处理脚本。

### 训练脚本
- **GBDT_rollingtrain.py**：单独训练所有截面模型的主文件，支持输入参数time period 可以指定滚动训练的年份。
- **GRU_GBDT_rollingtrain.py**：滚动训练时序模型和截面模型的主文件，支持输入参数time period 可以指定滚动训练的年份。
- **GRU_train_ICloss_new.py**：GRU模型的IC损失训练脚本。
- **models_train_win.py**：Windows平台专用的模型训练脚本。

### 数据处理模块
- **get_data**：数据加载模块，不同因子库单独处理。
  - CCB/：CCB数据源处理
  - CY312/：CY312数据源处理（包含跨截面和时序数据加载器）
  - Ding128/：Ding128数据源处理
  - DrJin129/：DrJin129数据源处理
  - minute_factors/：分钟级因子处理
  - data_align_and_save/：数据对齐和保存工具

### 评估和验证模块
- **metrics**：评估模块，包含：
  - **backtest.py**：回测分析
    - **分档回测**：将股票按预测值分成10档进行回测分析
    - **收益率计算**：计算各档位的累计收益率和年化收益率
    - **风险指标**：
      - 最大回撤(Max Drawdown)：衡量投资组合的最大损失
      - 夏普比率(Sharpe Ratio)：衡量风险调整后的收益
      - 卡玛比率(Calmar Ratio)：年化收益/最大回撤
      - 波动率(Volatility)：收益率的标准差
    - **交易成本**：
      - 双边换手率：买入和卖出的总换手率
      - 交易费用计算：考虑手续费和冲击成本
    - **绩效分析**：
      - 信息比率(Information Ratio)：超额收益/跟踪误差
      - 胜率：正收益期数占比
      - 收益分布：各期收益的统计分析
    - **可视化**：生成收益曲线图、回撤图、分档收益对比图
  - **calculate_ic.py**：IC计算
    - Information Coefficient：衡量预测值与实际收益的相关性
    - IC统计：IC均值、IC标准差、IC_IR、胜率等
    - 分组IC：按行业、市值、时间等维度计算IC
    - IC衰减分析：不同预测期限的IC表现
  - **factor_validator.py**：**新增** 因子验证器
    - 因子有效性检验：单调性、稳定性、显著性测试
    - 因子覆盖度：有效数据占比分析
    - 因子分布：异常值检测和分布分析
    - 因子相关性：与已有因子的相关性分析
  - **models_pred.py**：模型预测评估
    - 多模型预测结果对比
    - 模型融合策略
    - 预测准确度评估
  - **log.py**：日志记录
    - 训练过程日志
    - 错误信息记录
    - 性能监控日志
  - **train_plot.py**：训练过程可视化
    - 损失函数曲线：训练集和验证集loss变化
    - IC曲线：验证集IC随训练轮次的变化
    - 学习率变化：学习率调度可视化
    - 梯度范数：监控梯度爆炸/消失问题

### 模型模块
- **model**：模型模块，包含：
  - **Encoder.py**：**更新** 包含TimeSeriesEncoder等编码器
    - TimeSeriesEncoder：专为时间序列数据设计的编码器，支持多种时间特征提取
    - 支持多头注意力机制和位置编码
    - 可配置的层数和隐藏维度
  - **GRU_attention.py**：注意力机制GRU
    - 结合GRU和注意力机制，提升长序列建模能力
    - 支持自注意力和交叉注意力机制
    - 可处理变长序列输入
  - **GRU_model.py**：基础GRU模型
    - 标准GRU实现，支持双向和单向模式
    - 可配置层数、隐藏维度和dropout
    - 适合处理时间序列预测任务
  - **TimeMixer.py**：TimeMixer模型
    - 先进的时间序列预测模型，结合卷积和注意力机制
    - 支持多尺度时间特征提取
    - 高效的长序列建模能力
  - **lgb_pred.py**：LightGBM预测模块
    - 集成LightGBM、XGBoost、CatBoost等树模型
    - 支持超参数自动调优
    - 提供特征重要性分析
  - **losses.py**：损失函数
    - IC损失：针对金融预测的Information Coefficient损失
    - Rank损失：排序相关的损失函数
    - 支持自定义损失函数
  - **layers/**：模型层组件
    - Autoformer_EncDec.py：Autoformer编码解码层
    - Embed.py：嵌入层实现
    - StandardNorm.py：标准化层

### 训练模块
- **train**：训练模块，包含：
  - GBDT_trainer.py：GBDT训练器
  - GRU_cross_time_train.py：GRU跨时间训练
  - trainer.py：通用训练器

### 工具模块
- **utils**：工具模块，包含：
  - data_analysis.py：数据分析
  - fillna.py：缺失值处理
  - rolling.py：**新增** 滚动窗口处理工具
  - time_parser.py：**新增** 时间解析工具
  - masking.py：数据掩码
  - metrics.py：评估指标
  - timefeatures.py：时间特征工程
  - tools.py：通用工具函数

### 演示脚本
- **models_train_demo.sh**：模型训练演示脚本
- **models_pred_demo.sh**：模型预测演示脚本  
- **trees_train_demo.sh**：树模型训练演示脚本

## 开始使用

### 先决条件

- Python 3.10.17
- PyTorch 2.6.0+cu118
- LightGBM 4.6.0
- XGBoost 3.0.0
- CatBoost 1.2.8

### 安装

```bash
pip install -r requirements.txt
```

### 使用

#### 1. 数据加载

使用get_data模块中的函数加载数据。根据不同的因子存储格式分别使用不同的data_align_and_save函数，最终转存成统一格式：不同年份分开储存，每个年份文件夹下每个因子储存成[B,T]大小的内外样本parquet

**针对存储格式为每个因子单独储存的pickle：**
```bash
python get_data/data_align_and_save/save_CY312.py
```

**针对存储格式为[C,B,T]的npy文件：**
```bash
python get_data/data_align_and_save/save_DrJin129_row.py
python get_data/data_align_and_save/save_DrJin129_Rolling.py
```

**针对分钟频率因子：**
```bash
python get_data/data_align_and_save/save_minute_features.py
python get_data/data_align_and_save/rename_minute.py
```

#### 2. 配置实验参数

在config.py文件中修改配置实验参数，如任务名称、因子类型、模型类型、模型参数等。

#### 3. 模型训练

**并行训练多个滚动训练任务：**
```bash
bash models_train_demo.sh
```

**单独训练不同类型模型：**
```bash
# 时序模型训练
python GRU_train_ICloss_new.py

# 截面模型训练  
python GBDT_rollingtrain.py

# Windows平台训练
python models_train_win.py
```

**树模型专用训练：**
```bash
bash trees_train_demo.sh
```

#### 4. 模型预测

**批量预测：**
```bash
bash models_pred_demo.sh
```

**单独预测：**
```bash
python fintest_savepred.py
```

#### 5. 滚动预测处理

```bash
python rolling_pred_concat.py
```

## 模型架构详解

### 时序模型
**GRU系列模型**
- **基础GRU**: 处理时间序列依赖关系，支持多层堆叠
- **BiGRU**: 双向GRU，同时利用前向和后向信息
- **AttGRU**: 注意力增强GRU，自动学习重要时间步的权重

**Transformer系列模型**
- **TimeMixer**: 混合卷积和自注意力机制，高效处理长序列
- **TimeSeriesEncoder**: 专为金融时序设计的编码器
  - 位置编码：处理时间序列的位置信息
  - 多头注意力：捕获不同时间尺度的特征
  - 残差连接：解决深层网络退化问题

### 截面模型
**树模型集成**
- **LightGBM**: 快速梯度提升框架，支持GPU加速
- **XGBoost**: 极端梯度提升，内置正则化
- **CatBoost**: 处理类别特征友好，无需预处理

**模型融合策略**
- 加权平均：基于验证集IC表现的动态权重
- Stacking：使用元学习器进行二级融合
- Voting：多数投票机制

## 回测系统详解

### 分档回测机制
**分档策略**
- 按预测值将股票分成10档（或可配置档数）
- 每档包含相等数量的股票
- 支持行业中性化分档

**持仓构建**
- 多头策略：买入第1档（预测收益最高）
- 多空策略：买入第1档，卖空第10档
- 权重分配：等权重或按预测值加权

### 绩效指标体系
**收益指标**
- 累计收益率：总的投资收益
- 年化收益率：按年计算的平均收益
- 超额收益：相对基准的收益差异
- 分档收益分布：各档位的收益统计

**风险指标**
- 最大回撤：从峰值到谷值的最大跌幅
- 回撤持续期：回撤恢复到峰值的时间
- VaR：风险价值，一定置信度下的最大损失
- 下行波动率：只计算负收益的波动

**风险调整收益**
- 夏普比率：(年化收益 - 无风险收益) / 年化波动率
- 索提诺比率：年化收益 / 下行波动率
- 卡玛比率：年化收益 / 最大回撤
- 信息比率：年化超额收益 / 跟踪误差

**交易成本分析**
- 换手率：每期调仓的股票比例
- 交易费用：手续费 + 印花税 + 冲击成本
- 净收益：扣除交易成本后的收益
- 容量分析：策略可承受的资金规模

### IC分析框架
**IC统计指标**
- IC均值：平均Information Coefficient
- IC标准差：IC的波动性
- IC_IR：IC均值 / IC标准差
- IC胜率：IC > 0的期数占比
- IC分布：IC值的直方图分析

**分层IC分析**
- 行业IC：各行业内的IC表现
- 市值IC：不同市值分组的IC
- 时间IC：不同时间段的IC稳定性
- 因子IC：各个因子的贡献度分析

## 新增功能

### 因子验证器 (Factor Validator)
- 路径：`metrics/factor_validator.py`
- 功能：提供因子有效性验证、质量评估等功能

### 滚动预测处理
- 路径：`rolling_pred_concat.py`
- 功能：处理和连接多个滚动预测结果

### 跨平台支持
- Windows专用训练脚本：`models_train_win.py`
- 支持Windows环境下的模型训练

### 增强工具模块
- **时间解析工具** (`utils/time_parser.py`)：提供时间序列解析功能
- **滚动处理工具** (`utils/rolling.py`)：提供滚动窗口处理功能

### 模型增强
- **TimeSeriesEncoder**：新增的时间序列编码器模型
- 改进的模型训练和预测流程

## 注意事项

1. **数据格式**：确保数据按照项目要求的格式进行预处理
2. **依赖管理**：建议使用虚拟环境管理依赖
3. **GPU支持**：部分模型需要GPU支持，确保CUDA环境正确配置
4. **内存使用**：大规模数据训练时注意内存使用情况
5. **因子扩展**：增加因子时需要修改对应的load_x_parquet函数

## 项目结构图

```
Finforecast/
├── config.py                    # 总配置文件
├── fintest_savepred.py          # 模型推理主文件
├── rolling_pred_concat.py       # 滚动预测处理
├── models_train_win.py          # Windows训练脚本
├── get_data/                    # 数据加载模块
├── metrics/                     # 评估模块
│   └── factor_validator.py     # 因子验证器
├── model/                       # 模型模块
├── train/                       # 训练模块
├── utils/                       # 工具模块
│   ├── rolling.py              # 滚动处理工具
│   └── time_parser.py          # 时间解析工具
└── *.sh                        # 演示脚本
```
   
   