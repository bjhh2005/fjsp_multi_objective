"""
结果保存工具
"""

import pickle
import csv
import numpy as np
from typing import List, Dict, Any

def save_pickle_results(results, filepath):
    """保存pickle格式结果"""
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

def save_text_results(results, filepath):
    """保存文本格式的结果"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("柔性作业车间调度多目标优化结果\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本信息
        f.write(f"算法运行时间: {results.get('runtime', 0):.2f} 秒\n")
        f.write(f"Pareto前沿大小: {len(results.get('objectives', []))}\n")
        f.write(f"总迭代次数: {len(results.get('generation_history', []))}\n\n")
        
        # Pareto前沿
        f.write("Pareto前沿解集:\n")
        f.write("-" * 30 + "\n")
        objectives = results.get('objectives', [])
        for i, obj in enumerate(objectives):
            f.write(f"解 {i+1}: Makespan = {obj[0]:.2f}, Workload = {obj[1]:.2f}\n")
        
        # 统计信息
        if objectives:
            makespans = [obj[0] for obj in objectives]
            workloads = [obj[1] for obj in objectives]
            
            f.write("\n统计信息:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Makespan - 最小值: {min(makespans):.2f}, 最大值: {max(makespans):.2f}, 平均值: {np.mean(makespans):.2f}\n")
            f.write(f"Workload - 最小值: {min(workloads):.2f}, 最大值: {max(workloads):.2f}, 平均值: {np.mean(workloads):.2f}\n")

def save_csv_results(results, filepath):
    """保存CSV格式的Pareto前沿"""
    objectives = results.get('objectives', [])
    if not objectives:
        return
        
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Solution_ID', 'Makespan', 'Total_Workload'])
        
        for i, obj in enumerate(objectives):
            writer.writerow([i+1, obj[0], obj[1]])

def save_best_solution_details(best_solution, best_schedule, best_makespan, 
                             best_workload, selection_criteria, filepath):
    """保存最优解的详细信息"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("最优调度方案详情\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"选择标准: {selection_criteria}\n")
        f.write(f"目标函数值:\n")
        f.write(f"  - 总流程时间 (Makespan): {best_makespan:.2f}\n")
        f.write(f"  - 机器总负载 (Workload): {best_workload:.2f}\n\n")
        
        if best_schedule:
            f.write("调度方案:\n")
            f.write("-" * 20 + "\n")
            
            # 按机器分组显示
            machine_operations = {}
            for op in best_schedule:
                machine_id = op['machine_id']
                if machine_id not in machine_operations:
                    machine_operations[machine_id] = []
                machine_operations[machine_id].append(op)
            
            # 按机器顺序输出
            for machine_id in sorted(machine_operations.keys()):
                f.write(f"\n机器 {machine_id}:\n")
                operations = sorted(machine_operations[machine_id], key=lambda x: x['start_time'])
                for op in operations:
                    f.write(f"  工序 J{op['job_id']}-O{op['operation_id']}: ")
                    f.write(f"开始时间 {op['start_time']:.1f} -> 结束时间 {op['finish_time']:.1f} ")
                    f.write(f"(加工时间 {op['processing_time']:.1f})\n")
            
            # 机器负载统计
            f.write("\n机器负载统计:\n")
            f.write("-" * 20 + "\n")
            machine_workloads = {}
            for op in best_schedule:
                machine_id = op['machine_id']
                machine_workloads[machine_id] = machine_workloads.get(machine_id, 0) + op['processing_time']
            
            for machine_id in sorted(machine_workloads.keys()):
                workload = machine_workloads[machine_id]
                utilization = (workload / best_makespan) * 100 if best_makespan > 0 else 0
                f.write(f"机器 {machine_id}: 负载 {workload:.2f}, 利用率 {utilization:.1f}%\n")
        else:
            f.write("无法生成调度方案详情: 最优解解码失败\n")

def save_all_results(results, decoder, best_solution, best_schedule, 
                   best_makespan, best_workload, selection_criteria, 
                   algorithm_name, output_dir):
    """保存所有结果文件"""
    import os
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 保存pickle格式
        save_pickle_results(results, f'{output_dir}/{algorithm_name}_results.pkl')
        
        # 保存文本格式
        save_text_results(results, f'{output_dir}/{algorithm_name}_results.txt')
        
        # 保存CSV格式
        save_csv_results(results, f'{output_dir}/{algorithm_name}_pareto_front.csv')
        
        # 保存最优解详情
        save_best_solution_details(best_solution, best_schedule, best_makespan,
                                 best_workload, selection_criteria,
                                 f'{output_dir}/{algorithm_name}_best_solution.txt')
        
        return True, "所有结果保存成功"
        
    except Exception as e:
        return False, f"保存结果失败: {str(e)}"