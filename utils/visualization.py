"""
可视化工具
用于绘制Pareto前沿和甘特图
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.font_manager as fm

# 设置中文字体
try:
    # 尝试使用系统中已有的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 中文字体设置失败，图表可能无法正确显示中文")

def plot_pareto_front(objectives_list, algorithm_names, save_path=None):
    """绘制多个算法的Pareto前沿"""
    plt.figure(figsize=(10, 8))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*']
    
    for i, (objectives, name) in enumerate(zip(objectives_list, algorithm_names)):
        if not objectives:
            continue
            
        makespans = [obj[0] for obj in objectives]
        workloads = [obj[1] for obj in objectives]
        
        plt.scatter(makespans, workloads, c=colors[i % len(colors)], 
                    marker=markers[i % len(markers)], label=name, alpha=0.7, s=50)
    
    plt.xlabel('Makespan')
    plt.ylabel('Total Workload')
    plt.title('Pareto Front Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_gantt_chart(schedule, save_path=None):
    """绘制甘特图 - 适配新的schedule结构"""
    if not schedule:
        print("Empty schedule!")
        return
    
    try:
        # 检查schedule的数据结构
        print(f"Schedule type: {type(schedule)}")
        if isinstance(schedule, list) and len(schedule) > 0:
            print(f"First element type: {type(schedule[0])}")
            print(f"First element: {schedule[0]}")
        
        # 适配不同的schedule结构
        if isinstance(schedule, list) and len(schedule) > 0:
            # 新的字典结构
            if isinstance(schedule[0], dict):
                operations = schedule
            # 旧的元组结构
            elif isinstance(schedule[0], (tuple, list)) and len(schedule[0]) >= 5:
                operations = [
                    {
                        'job_id': op[0],
                        'operation_id': op[1],
                        'machine_id': op[2],
                        'start_time': op[3],
                        'finish_time': op[4]
                    }
                    for op in schedule
                ]
            else:
                raise ValueError(f"不支持的schedule结构: {type(schedule[0])}")
        else:
            raise ValueError("无效的schedule数据")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 获取所有机器ID
        machine_ids = sorted(set(op['machine_id'] for op in operations))
        num_machines = len(machine_ids)
        
        if num_machines == 0:
            print("没有找到机器信息")
            return
        
        # 为每个机器创建y轴位置
        machine_ypos = {machine_id: i for i, machine_id in enumerate(machine_ids)}
        
        # 为每个工件创建颜色
        job_ids = sorted(set(op['job_id'] for op in operations))
        colors = plt.cm.tab10(np.linspace(0, 1, len(job_ids)))
        job_colors = {job_id: colors[i % len(colors)] for i, job_id in enumerate(job_ids)}
        
        # 绘制每个工序
        for op in operations:
            job_id = op['job_id']
            op_id = op['operation_id']
            machine_id = op['machine_id']
            start_time = op['start_time']
            finish_time = op['finish_time']
            
            y_pos = machine_ypos[machine_id]
            duration = finish_time - start_time
            
            # 绘制矩形
            ax.barh(y_pos, duration, left=start_time, 
                    height=0.6, color=job_colors[job_id], alpha=0.7,
                    edgecolor='black', linewidth=0.5)
            
            # 添加标签
            label = f'J{job_id}-O{op_id}'
            ax.text(start_time + duration/2, y_pos, label, 
                    ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        
        # 设置坐标轴
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f'Machine {mid}' for mid in machine_ids])
        ax.set_xlabel('Time')
        ax.set_ylabel('Machines')
        ax.set_title('Gantt Chart')
        
        # 设置网格和布局
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"甘特图已保存: {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"绘制甘特图时出错: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")

def plot_convergence(objectives_history, algorithm_names, save_path=None):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 8))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (history, name) in enumerate(zip(objectives_history, algorithm_names)):
        if not history:
            continue
            
        # 计算每代的最佳目标值
        best_makespans = [min([obj[0] for obj in gen]) for gen in history if gen]
        best_workloads = [min([obj[1] for obj in gen]) for gen in history if gen]
        
        # 绘制makespan收敛曲线
        plt.subplot(2, 1, 1)
        plt.plot(best_makespans, 
                c=colors[i % len(colors)], 
                linestyle=linestyles[i % len(linestyles)],
                label=name, linewidth=2)
        
        # 绘制workload收敛曲线
        plt.subplot(2, 1, 2)
        plt.plot(best_workloads, 
                c=colors[i % len(colors)], 
                linestyle=linestyles[i % len(linestyles)],
                label=name, linewidth=2)
    
    plt.subplot(2, 1, 1)
    plt.xlabel('Generation')
    plt.ylabel('Best Makespan')
    plt.title('Convergence - Makespan')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Generation')
    plt.ylabel('Best Workload')
    plt.title('Convergence - Workload')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_algorithm_comparison(results_dict, save_path=None):
    """绘制算法比较图"""
    algorithms = list(results_dict.keys())
    
    # 提取指标
    makespans = [min(obj[0] for obj in results_dict[alg]['objectives']) for alg in algorithms]
    workloads = [min(obj[1] for obj in results_dict[alg]['objectives']) for alg in algorithms]
    runtimes = [results_dict[alg].get('runtime', 0) for alg in algorithms]
    pareto_sizes = [len(results_dict[alg]['objectives']) for alg in algorithms]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Makespan比较
    bars1 = axes[0, 0].bar(algorithms, makespans, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Best Makespan Comparison')
    axes[0, 0].set_ylabel('Makespan')
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
    
    # 2. Workload比较
    bars2 = axes[0, 1].bar(algorithms, workloads, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Best Workload Comparison')
    axes[0, 1].set_ylabel('Workload')
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
    
    # 3. 运行时间比较
    bars3 = axes[1, 0].bar(algorithms, runtimes, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Runtime Comparison')
    axes[1, 0].set_ylabel('Runtime (seconds)')
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}s', ha='center', va='bottom')
    
    # 4. Pareto前沿大小比较
    bars4 = axes[1, 1].bar(algorithms, pareto_sizes, color='gold', alpha=0.7)
    axes[1, 1].set_title('Pareto Front Size Comparison')
    axes[1, 1].set_ylabel('Number of Solutions')
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()