"""
可视化工具
用于绘制Pareto前沿和甘特图
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_pareto_front(objectives_list, algorithm_names, save_path=None):
    """绘制多个算法的Pareto前沿"""
    plt.figure(figsize=(10, 8))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*']
    
    for i, (objectives, name) in enumerate(zip(objectives_list, algorithm_names)):
        makespans = [obj[0] for obj in objectives]
        workloads = [obj[1] for obj in objectives]
        
        plt.scatter(makespans, workloads, c=colors[i % len(colors)], 
                    marker=markers[i % len(markers)], label=name, alpha=0.7)
    
    plt.xlabel('Makespan (总流程时间)')
    plt.ylabel('Total Workload (机器总负载)')
    plt.title('Pareto Front Comparison (Pareto前沿比较)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_gantt_chart(schedule, save_path=None):
    """绘制甘特图"""
    if not schedule:
        print("Empty schedule!")
        return
    
    # 提取所有工序
    operations = sorted(schedule, key=lambda x: (x[3], x[0]))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 为每个机器创建一个颜色
    machine_colors = {}
    for job_id, op_id, machine_id, start_time, finish_time in schedule:
        if machine_id not in machine_colors:
            # 修复：使用matplotlib.colors而不是matplotlib.cm.tab10
            machine_colors[machine_id] = plt.cm.tab10(machine_id % 10)
    
    # 绘制每个工序
    for job_id, op_id, machine_id, start_time, finish_time in schedule:
        # 绘制矩形
        ax.barh(machine_id, finish_time - start_time, left=start_time, 
                height=0.8, color=machine_colors[machine_id], alpha=0.7)
        
        # 添加标签
        ax.text((start_time + finish_time) / 2, machine_id, 
                f'J{job_id}-O{op_id}', 
                ha='center', va='center', color='white', fontsize=8)
    
    # 设置坐标轴
    ax.set_yticks(range(1, max(machine_colors.keys()) + 1))
    ax.set_yticklabels([f'Machine {i}' for i in range(1, max(machine_colors.keys()) + 1)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Gantt Chart (甘特图)')
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_convergence(objectives_history, algorithm_names, save_path=None):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    for i, (history, name) in enumerate(zip(objectives_history, algorithm_names)):
        # 计算每代的平均目标值
        avg_makespans = [np.mean([obj[0] for obj in gen]) for gen in history]
        avg_workloads = [np.mean([obj[1] for obj in gen]) for gen in history]
        
        # 绘制makespan收敛曲线
        plt.subplot(2, 1, 1)
        plt.plot(avg_makespans, c=colors[i % len(colors)], label=f'{name} - Makespan')
        
        # 绘制workload收敛曲线
        plt.subplot(2, 1, 2)
        plt.plot(avg_workloads, c=colors[i % len(colors)], label=f'{name} - Workload')
    
    plt.subplot(2, 1, 1)
    plt.xlabel('Generation')
    plt.ylabel('Average Makespan')
    plt.title('Convergence Curves - Makespan')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Generation')
    plt.ylabel('Average Workload')
    plt.title('Convergence Curves - Workload')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()