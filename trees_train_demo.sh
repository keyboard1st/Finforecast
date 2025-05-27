#!/bin/bash

#nohup python -u /home/hongkou/TimeSeries/GBDT_rollingtrain.py --task_name 'CY_2021_2022' --time_period '2021-2022' --device 'cuda:1' > GBDT_train_2021_2022.log 2>&1 &
nohup python -u /home/hongkou/TimeSeries/GBDT_rollingtrain.py --task_name 'CY_2022_2023' --time_period '2022-2023' --device 'cuda:3' --early_stop_patience 2 > GBDT_train_2022_2023.log 2>&1 &
nohup python -u /home/hongkou/TimeSeries/GBDT_rollingtrain.py --task_name 'CY_2023_2024' --time_period '2023-2024' --device 'cuda:7' --num_val_windows 300 --early_stop_patience 2 > GBDT_train_2023_2024.log 2>&1 &