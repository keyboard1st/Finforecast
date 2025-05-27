#!/bin/bash

nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'Jin_2021_2022_randomseed2025' --time_period '2021-2022' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2021_2022_randomseed2025' --device 'cuda:1' --random_seed 2025 > Ztest_2021_2022.log 2>&1 &
nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'Jin_2022_2023_randomseed2025' --time_period '2022-2023' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2022_2023_randomseed2025' --device 'cuda:2' --random_seed 2025 > Ztest_2022_2023.log 2>&1 &
nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'Jin_2023_2024_randomseed2025' --time_period '2023-2024' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2023_2024_randomseed2025' --device 'cuda:3' --random_seed 2025 > Ztest_2023_2024.log 2>&1 &
nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'Jin_2024_2025_randomseed2025' --time_period '2024-2025' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2024_2025_randomseed2025' --device 'cuda:4' --random_seed 2025 > Ztest_2024_2025.log 2>&1 &

##nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2021_2022_std' --time_period '2021-2022' --device 'cuda:0' > test_2021_2022.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2022_2023_std' --time_period '2022-2023' --device 'cuda:3' --early_stop_patience 2 > test_2022_2023.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2023_2024_std' --time_period '2023-2024' --device 'cuda:7' --early_stop_patience 2 > test_2023_2024.log 2>&1 &

#nohup python -u /home/hongkou/TimeSeries/GRU_train_ICloss_new.py --task_name 'minute10_2021_2022' --time_period '2021-2022' --model_type 'TimeMixer' --device 'cuda:5' --early_stop_patience 3 --loss 'MSE' > ZMtest_2021_2022.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_train_ICloss_new.py --task_name 'minute10_2022_2023' --time_period '2022-2023' --model_type 'TimeMixer' --device 'cuda:6' --early_stop_patience 3 --loss 'MSE' > ZMtest_2022_2023.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_train_ICloss_new.py --task_name 'minute10_2023_2024' --time_period '2023-2024' --model_type 'TimeMixer' --device 'cuda:7' --early_stop_patience 3 --loss 'MSE' > ZMtest_2023_2024.log 2>&1 &