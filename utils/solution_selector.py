"""
最优解选择工具
"""

import numpy as np
from typing import Tuple, Any, List, Dict

def get_best_solution(results, decoder, selection_criteria='makespan') -> Tuple[Any, Any, float, float, str]:
    """
    获取最优解
    
    Args:
        results: 算法结果
        decoder: 解码器
        selection_criteria: 选择标准 ('makespan', 'workload', 'balanced')
    
    Returns:
        best_solution: 最优解染色体
        best_schedule: 最优解调度方案
        best_makespan: 最优解makespan
        best_workload: 最优解workload
        criteria_desc: 选择标准描述
    """
    pareto_front = results.get('pareto_front', [])
    objectives = results.get('objectives', [])
    
    if not pareto_front or not objectives:
        return None, None, float('inf'), float('inf'), "无有效解"
    
    if selection_criteria == 'makespan':
        # 选择makespan最小的解
        best_idx = np.argmin([obj[0] for obj in objectives])
        criteria_desc = "最小Makespan"
    elif selection_criteria == 'workload':
        # 选择workload最小的解
        best_idx = np.argmin([obj[1] for obj in objectives])
        criteria_desc = "最小Workload"
    else:  # balanced
        # 选择两个目标都相对较好的解（归一化后的加权和最小）
        best_idx, criteria_desc = _select_balanced_solution(objectives)
    
    best_solution = pareto_front[best_idx]
    best_objectives = objectives[best_idx]
    
    # 解码最优解
    try:
        schedule, makespan, workload = decoder.decode(best_solution)
        return best_solution, schedule, makespan, workload, criteria_desc
    except Exception as e:
        print(f"解码最优解时出错: {e}")
        return best_solution, None, best_objectives[0], best_objectives[1], criteria_desc

def _select_balanced_solution(objectives: List[List[float]]) -> Tuple[int, str]:
    """选择平衡的解"""
    makespans = [obj[0] for obj in objectives]
    workloads = [obj[1] for obj in objectives]
    
    # 归一化
    norm_makespans = (makespans - np.min(makespans)) / (np.max(makespans) - np.min(makespans) + 1e-8)
    norm_workloads = (workloads - np.min(workloads)) / (np.max(workloads) - np.min(workloads) + 1e-8)
    
    # 计算综合得分（越小越好）
    scores = norm_makespans + norm_workloads
    best_idx = np.argmin(scores)
    
    return best_idx, "平衡解(Makespan和Workload综合最优)"

def get_selection_criteria_description(criteria: str) -> str:
    """获取选择标准的描述"""
    descriptions = {
        'makespan': '最小总流程时间',
        'workload': '最小机器总负载', 
        'balanced': '平衡解(两个目标综合最优)'
    }
    return descriptions.get(criteria, '未知标准')