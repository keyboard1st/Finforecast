#!/bin/bash

nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'Jin_2021_2022_randomseed0' --time_period '2021-2022' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2021_2022_randomseed0' --device 'cuda:1' --random_seed 0 > test_2021_2022.log 2>&1 &
nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'Jin_2022_2023_randomseed0' --time_period '2022-2023' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2022_2023_randomseed0' --device 'cuda:2' --random_seed 0 > test_2022_2023.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'Jin_2023_2024_randomseed0' --time_period '2023-2024' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2023_2024_randomseed0' --device 'cuda:3' --random_seed 0 > test_2023_2024.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'Jin_2024_2025_randomseed0' --time_period '2024-2025' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2024_2025_randomseed0' --device 'cuda:4' --random_seed 0 > test_2024_2025.log 2>&1 &

##nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2021_2022_std' --time_period '2021-2022' --device 'cuda:0' > test_2021_2022.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2022_2023_std' --time_period '2022-2023' --device 'cuda:3' --early_stop_patience 2 > test_2022_2023.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2023_2024_std' --time_period '2023-2024' --device 'cuda:7' --early_stop_patience 2 > test_2023_2024.log 2>&1 &
