#!/bin/bash

nohup python D:\chenxing\Finforecast\GRU_GBDT_rollingtrain.py --task_name 'AttGRU_244_202404_202406' --train_time_period '201901-202403' --test_time_period '202404-202406' > test1.log 2>&1 &
nohup python D:\chenxing\Finforecast\GRU_GBDT_rollingtrain.py --task_name 'AttGRU_244_202407_202409' --train_time_period '201901-202406' --test_time_period '202407-202409' > test2.log 2>&1 &
nohup python D:\chenxing\Finforecast\GRU_GBDT_rollingtrain.py --task_name 'AttGRU_244_202410_202412' --train_time_period '201901-202409' --test_time_period '202410-202412' > test3.log 2>&1 &
nohup python D:\chenxing\Finforecast\GRU_GBDT_rollingtrain.py --task_name 'AttGRU_244_202501_202504' --train_time_period '201901-202412' --test_time_period '202501-202504' > test4.log 2>&1 &

##nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2021_2022_std' --time_period '2021-2022' --device 'cuda:0' > test_2021_2022.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2022_2023_std' --time_period '2022-2023' --device 'cuda:3' --early_stop_patience 2 > test_2022_2023.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2023_2024_std' --time_period '2023-2024' --device 'cuda:7' --early_stop_patience 2 > test_2023_2024.log 2>&1 &

#nohup python -u /home/hongkou/TimeSeries/GRU_train_ICloss_new.py --task_name 'minute10_2021_2022' --time_period '2021-2022' --model_type 'TimeMixer' --device 'cuda:5' --early_stop_patience 3 --loss 'MSE' > ZMtest_2021_2022.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_train_ICloss_new.py --task_name 'minute10_2022_2023' --time_period '2022-2023' --model_type 'TimeMixer' --device 'cuda:6' --early_stop_patience 3 --loss 'MSE' > ZMtest_2022_2023.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_train_ICloss_new.py --task_name 'minute10_2023_2024' --time_period '2023-2024' --model_type 'TimeMixer' --device 'cuda:7' --early_stop_patience 3 --loss 'MSE' > ZMtest_2023_2024.log 2>&1 &