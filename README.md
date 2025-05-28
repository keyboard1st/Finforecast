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
- train：训练模块，包含多个子模块，如GBDT_trainer.py、GRU_cross_time_train.py等。
- utils：工具模块，包含多个子模块，如fillna.py、m4_summary.py、masking.py、metrics.py、timefeatures.py、tools.py等。
- GBDT_rollingtrain.py：单独训练所有截面模型的主文件，支持输入参数time period 可以指定滚动训练的年份。
- GRU_GBDT_rollingtrain.py：滚动训练时序模型和截面模型的主文件，支持输入参数time period 可以指定滚动训练的年份。

## 开始使用

### 先决条件

- Python 3.10.17
- PyTorch 2.6.0+cu118
- LightGBM 4.6.0
- XGBoost 3.0.0
- CatBoost 1.2.8

### 安装

```
pip install -r requirements.txt
```

### 使用

1. 数据加载：使用get_data模块中的函数加载数据。
   根据不同的因子存储格式分别使用不同的data_align_and_save函数，最终转存成统一格式：不同年份分开储存，每个年份文件夹下每个因子储存成[B,T]大小的内外样本parquet

   - 针对存储格式为每个因子单独储存的pickle，每个pickle大小为[B,T]，修改路径后运行
      ```
      python save_CY312.py
      ```
   - 针对存储格式为[C,B,T]的npy文件，文件中包含C个因子，每个因子都为大小为[B,T]的tensor，并且列名和索引名分开存储的格式，修改路径后运行
   - 注意：如果因子分成两种类型：‘利用交易时间之前计算的因子值’和‘利用全天数据计算的因子值’，下面的脚本需要运行两次，后续时序loader中自动读取不同类型的因子拼接成时间窗口，来防止数据泄露
      ```
      python save_DrJin129_row.py
      python save_DrJin129_Rolling.py
      ```
   - 针对存储格式为[B,T]的npy文件，不同因子单独存储的分钟频率因子，修改路径后运行
      ```
      python save_minute_features.py
      python rename_minute.py # 用于重命名因子、并且保存因子名称映射表
      ```
   - 最后不同因子的dataset和dataloader分别保存到不同的文件夹下，可以通过调用对应因子名文件夹下的get_loader()直接使用，后续因子时间段延长时，只需要运行上面的程序更新就可以了，不需要更改get_loader
   - 如果需要增加因子，需要修改对应文件夹下的load_x_parquet函数中，增加文件名筛选范围即可，例如因子个数从129增加到135，修改load_x_parquet函数中：path_list = path([1,130]) -> path([1,136])
   

2. 配置实验参数：在config.py文件中修改配置实验参数，如任务名称、因子类型、模型类型、模型参数等。
3. 模型训练：使用下面命令会并行训练多个滚动训练任务（会同时训练时序模型和树模型），训练保存的模型权重会保存到对应的任务文件夹 exp/{task_name} 下。
   ```
   bash models_train_demo.sh
   ```
   - 如果需要单独训练树模型或者单独训练时序模型，修改好路径后可以直接运行下面的脚本：
   ```
   python GRU_train_ICloss_new.py
   python GBDT_rollingtrain.py
   ```


4. 模型预测：使用fintest_savepred.py文件中的函数进行模型预测，会保存模型预测结果csv和png文件到对应的 exp/{task_name} 下。
   ```
   bash models_pred_demo.sh
   ```
   
   