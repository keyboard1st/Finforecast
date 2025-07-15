#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import time

# 定义要运行的参数组合
tasks = [
    {
        'task_name': 'AttGRU_112_202401_202403',
        'train_time_period': '201901-202312',
        'test_time_period': '202401-202403'
    },
    {
        'task_name': 'AttGRU_112_202404_202406', 
        'train_time_period': '201901-202403',
        'test_time_period': '202404-202406'
    },
    {
        'task_name': 'AttGRU_112_202407_202409',
        'train_time_period': '201901-202406', 
        'test_time_period': '202407-202409'
    },
    {
        'task_name': 'AttGRU_112_202410_202412',
        'train_time_period': '201901-202409',
        'test_time_period': '202410-202412'
    },
    {
        'task_name': 'AttGRU_112_202501_202504',
        'train_time_period': '201901-202412', 
        'test_time_period': '202501-202504'
    },
]

def run_task(task):
    """运行单个任务"""
    cmd = [
        'python', 'GRU_GBDT_rollingtrain.py',
        '--task_name', task['task_name'],
        '--train_time_period', task['train_time_period'],
        '--test_time_period', task['test_time_period']
    ]
    
    print(f"启动任务: {task['task_name']}")
    print(f"命令: {' '.join(cmd)}")
    
    # 后台运行，日志保存到文件
    log_file = f"log_{task['task_name']}.log"
    with open(log_file, 'w') as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    print(f"任务已启动，日志文件: {log_file}\n")

def main():
    """主函数"""
    print("开始批量运行任务...\n")
    
    for i, task in enumerate(tasks):
        print(f"=== 任务 {i+1}/{len(tasks)} ===")
        run_task(task)
        
        if i < len(tasks) - 1:
            print("等待30秒...")
            time.sleep(30)
    
    print("所有任务已启动！")
    print("查看任务状态: ps aux | grep GRU_GBDT_rollingtrain.py")
    print("查看日志: tail -f log_任务名.log")

if __name__ == "__main__":
    main()