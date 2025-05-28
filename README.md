# Finforecast

## 项目概述

Finforecast是一个用于金融alpha预测的机器学习模型库，包含了多种时序学习模型，如GRU、BiGRU、AttGRU、TimeMixer等，以及GBDT、XGBoost、CatBoost等截面模型。项目还提供了数据加载、模型训练、模型评估等功能。

## 功能

- get_data：支持从Parquet文件中预处理数据、加载数据，并且返回时序和截面训练及推理各自所需的dataloader。
- train：支持单个时序模型和截面模型的训练。
- metrics：支持计算IC（Information Coefficient），根据预测值进行回测(分十档，并计算第一组的累计年化收益、最大回撤、Sharpe Ratio、双边换手率等)，训练过程中画图指标变化(train loss,vali loss,vali ic,learning rate)。
- model：模型库，包括多种时序学习模型，如GRU、BiGRU、AttGRU、TimeMixer以及GBDT、XGBoost、CatBoost截面模型。

## 项目结构

项目包含以下目录：

- config.py：总配置文件，包含实验参数、模型参数、数据路径等，更换所有实验参数只需要修改这个文件。
- fintest_savepred.py：最终模型推理和保存预测结果的主文件。
- get_data：数据加载模块，不同因子库单独处理。
- metrics：评估模块，包含多个子模块，如backtest.py、calculate_ic.py、log.py、train_plot.py等。
- model：模型模块，包含多个子模块，如Encoder.py、GRU_attention.py、GRU_model.py、TimeMixer.py等。
- readme.md：README文件。
- train：训练模块，包含多个子模块，如GBDT_trainer.py、GRU_cross_time_train.py等。
- utils：工具模块，包含多个子模块，如fillna.py、m4_summary.py、masking.py、metrics.py、timefeatures.py、tools.py等。
- GBDT_rollingtrain.py：单独训练所有截面模型的主文件，支持输入参数time period 可以指定滚动训练的年份。
- GRU_GBDT_rollingtrain.py：滚动训练时序模型和截面模型的主文件，支持输入参数time period 可以指定滚动训练的年份。
- rolling_GBDT_train.sh：滚动训练GBDT模型训练的脚本，支持同时训练多个任务。
- rolling_train_demo.sh：GRU和GBDT模型联合训练的脚本，支持同时训练多个任务。

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

1. 数据加载：使用get_data模块中的函数加载数据。
   根据不同的因子存储格式分别使用
   ```bash
  pip install -r requirements.txt
  ```
3. 配置实验参数：在config.py文件中配置实验参数，如模型类型、输入维度、隐藏层维度等。
4. 模型训练：使用GRU_GBDT_rollingtrain.py模块中的函数训练模型。
5. 模型预测：使用fintest_savepred.py文件中的函数进行模型预测。
