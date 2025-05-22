#!/bin/bash
#
## 设置 Python 环境（根据你的情况选择合适的环境）
## source /path/to/your/venv/bin/activate  # 如果你使用虚拟环境
#
## 训练函数
#GRU_rollingtrain() {
#    local task_name=$1
#    local time_period=$2
#    echo "Starting training for task: $task_name with time period: $time_period"
#    python train.py --task_name "$task_name" \
#                    --model_type "some_model" \
#                    --input_dim 128 \
#                    --hidden_dim 64 \
#                    --num_layers 2 \
#                    --output_dim 10 \
#                    --learning_rate 0.001 \
#                    --early_stop_patience 10 \
#                    --train_epochs 100 \
#                    --num_val_windows 5 \
#                    --pct_start 0.1 \
#                    --lradj "linear" \
#                    --exp_path "/home/hongkou/chenx/exp/$task_name" \
#                    --time_period "$time_period" &
#}
#
## 定义不同的时间段和对应的 task_name
#time_periods=("2021-2022" "2022-2023" "2023-2024")
#task_names=("Jin_2021_2022" "Jin_2022_2023" "Jin_2023_2024")
#
## 并行训练
#for i in "${!time_periods[@]}"; do
#    task_name="${task_names[$i]}"
#    time_period="${time_periods[$i]}"
#    train_model "$task_name" "$time_period"
#done
#
## 等待所有后台任务完成
#wait
#echo "All training tasks are completed!"


#nohup python -u /home/hongkou/TimeSeries/GRU_rollingtrain.py --task_name 'Jin_2021_2022' --time_period '2021-2022' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2021_2022_new' --device 'cuda:0' --learning_rate 0.00005 --early_stop_patience 1 > test_2021_2022.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_rollingtrain.py --task_name 'Jin_2022_2023' --time_period '2022-2023' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2022_2023_new' --device 'cuda:1' > test_2022_2023.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_rollingtrain.py --task_name 'Jin_2023_2024' --time_period '2023-2024' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2023_2024_new' --device 'cuda:1' --num_val_windows 300 > test_2023_2024.log 2>&1 &
#nohup python -u /home/hongkou/TimeSeries/GRU_rollingtrain.py --task_name 'Jin_2024_2025' --time_period '2024-2025' --exp_path '/home/hongkou/TimeSeries/exp/Jin_2024_2025_new' --device 'cuda:7' --num_val_windows 300 > test_2024_2025.log 2>&1 &

#nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2021_2022_std' --time_period '2021-2022' --device 'cuda:0' > test_2021_2022.log 2>&1 &
nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2022_2023_std' --time_period '2022-2023' --device 'cuda:3' --early_stop_patience 2 > test_2022_2023.log 2>&1 &
nohup python -u /home/hongkou/TimeSeries/GRU_GBDT_rollingtrain.py --task_name 'CY_2023_2024_std' --time_period '2023-2024' --device 'cuda:7' --early_stop_patience 2 > test_2023_2024.log 2>&1 &
