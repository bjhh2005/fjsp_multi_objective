"""
解码器模块
将染色体解码为调度方案并计算目标值
"""

import numpy as np
from typing import List, Tuple, Dict, Any

class FJSPDecoder:
    """FJSP解码器 - 适配新的染色体结构"""
    
    def __init__(self, problem):
        self.problem = problem
        self.num_jobs = problem.num_jobs
        self.num_machines = problem.machines
        self.total_operations = problem.get_total_operations()
    
    def decode(self, chromosome):
        """
        解码染色体为调度方案
        新的染色体结构: {'sequence': [...], 'machines': {...}}
        """
        # 验证染色体结构
        if not isinstance(chromosome, dict) or 'sequence' not in chromosome or 'machines' not in chromosome:
            raise ValueError(f"无效的染色体结构: {type(chromosome)}")
        
        # 获取工序序列和机器分配
        sequence_part = chromosome['sequence']
        machines_dict = chromosome['machines']
        
        # 验证数据完整性
        if len(sequence_part) != self.total_operations:
            raise ValueError(f"工序序列长度不正确: 期望{self.total_operations}, 实际{len(sequence_part)}")
        
        # 计算调度方案
        schedule = self._calculate_schedule(sequence_part, machines_dict)
        
        # 计算目标值
        makespan, total_workload = self._calculate_objectives(schedule)
        
        return schedule, makespan, total_workload
    
    def _calculate_schedule(self, sequence_part: List, machines_dict: Dict) -> List[Tuple]:
        """计算调度方案 - 适配新染色体结构"""
        # 初始化机器状态 (1-based索引)
        machine_available_time = [0] * (self.num_machines + 1)
        
        # 初始化工件状态
        job_next_operation = [1] * (self.num_jobs + 1)  # 每个工件的下一个工序编号
        job_previous_finish = [0] * (self.num_jobs + 1)  # 每个工件上一个工序的完成时间
        
        schedule = []
        
        # 按工序序列顺序处理每个工序
        for position, job_id in enumerate(sequence_part):
            # 获取当前工件的下一个工序编号
            op_id = job_next_operation[job_id]
            job_next_operation[job_id] += 1
            
            # 获取工序对象
            operation = self.problem.jobs[job_id].operations[op_id - 1]
            
            # 从机器分配字典获取机器ID
            machine_key = (job_id, op_id - 1)  # op_id是1-based，但字典键是0-based
            if machine_key not in machines_dict:
                # 尝试其他可能的键格式
                machine_key = (job_id, op_id)
                if machine_key not in machines_dict:
                    raise KeyError(f"找不到工序({job_id}, {op_id})的机器分配")
            
            machine_id = machines_dict[machine_key] + 1  # 转换为1-based
            
            # 验证机器是否可用
            if machine_id not in operation.machines:
                # 如果分配的机器不可用，选择第一个可用机器
                available_machines = list(operation.machines.keys())
                machine_id = available_machines[0] if available_machines else 1
            
            # 获取加工时间
            processing_time = operation.machines.get(machine_id, 0)
            if processing_time == 0:
                # 如果加工时间为0，设置为一个默认值
                processing_time = 1
            
            # 计算开始时间
            machine_ready = machine_available_time[machine_id]
            job_ready = job_previous_finish[job_id]
            start_time = max(machine_ready, job_ready)
            finish_time = start_time + processing_time
            
            # 更新状态
            machine_available_time[machine_id] = finish_time
            job_previous_finish[job_id] = finish_time
            
            # 添加到调度方案
            schedule.append({
                'job_id': job_id,
                'operation_id': op_id,
                'machine_id': machine_id,
                'start_time': start_time,
                'finish_time': finish_time,
                'processing_time': processing_time,
                'position': position
            })
        
        return schedule
    
    def _calculate_objectives(self, schedule: List[Dict]) -> Tuple[float, float]:
        """计算目标值"""
        if not schedule:
            return float('inf'), float('inf')
        
        # 目标1: 最小化总流程时间 (makespan)
        makespan = max(op['finish_time'] for op in schedule)
        
        # 目标2: 最小化机器总负载
        machine_workload = {}
        for op in schedule:
            machine_id = op['machine_id']
            processing_time = op['processing_time']
            machine_workload[machine_id] = machine_workload.get(machine_id, 0) + processing_time
        
        total_workload = sum(machine_workload.values())
        
        return makespan, total_workload
    
    def validate_chromosome(self, chromosome) -> bool:
        """验证染色体结构是否有效"""
        try:
            # 基本结构检查
            if not isinstance(chromosome, dict):
                return False
            
            if 'sequence' not in chromosome or 'machines' not in chromosome:
                return False
            
            sequence = chromosome['sequence']
            machines = chromosome['machines']
            
            # 序列长度检查
            if len(sequence) != self.total_operations:
                return False
            
            # 机器分配完整性检查
            expected_operations = 0
            for job_id, job in enumerate(self.problem.jobs):
                for op_id in range(len(job.operations)):
                    if (job_id, op_id) not in machines:
                        return False
                    expected_operations += 1
            
            return expected_operations == self.total_operations
            
        except Exception:
            return False
    
    def get_schedule_info(self, schedule: List[Dict]) -> Dict[str, Any]:
        """获取调度方案的详细信息"""
        if not schedule:
            return {}
        
        makespan = max(op['finish_time'] for op in schedule)
        
        # 机器利用率
        machine_utilization = {}
        for op in schedule:
            machine_id = op['machine_id']
            if machine_id not in machine_utilization:
                machine_utilization[machine_id] = {
                    'total_workload': 0,
                    'operations': 0,
                    'utilization': 0.0
                }
            machine_utilization[machine_id]['total_workload'] += op['processing_time']
            machine_utilization[machine_id]['operations'] += 1
        
        # 计算利用率
        for machine_info in machine_utilization.values():
            machine_info['utilization'] = machine_info['total_workload'] / makespan if makespan > 0 else 0
        
        return {
            'makespan': makespan,
            'total_workload': sum(info['total_workload'] for info in machine_utilization.values()),
            'machine_utilization': machine_utilization,
            'schedule_length': len(schedule)
        }