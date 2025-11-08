"""
解码器模块
将染色体解码为调度方案并计算目标值
"""

import numpy as np

class FJSPDecoder:
    """FJSP解码器"""
    
    def __init__(self, problem):
        self.problem = problem
        self.num_jobs = problem.num_jobs
        self.num_machines = problem.machines
        self.total_operations = problem.get_total_operations()
    
    def decode(self, chromosome):
        """
        解码染色体为调度方案
        染色体结构: [机器分配部分, 工序排序部分]
        """
        # 分离机器分配和工序排序部分
        machine_part = chromosome[:self.total_operations]
        sequence_part = chromosome[self.total_operations:]
        
        # 解码机器分配（已经是有效的）
        machine_assignment = [int(m) + 1 for m in machine_part]  # 转换回1-based
        
        # 解码工序序列（使用更可靠的方法）
        operation_sequence = self._decode_sequence(sequence_part)
        
        # 计算调度方案
        schedule = self._calculate_schedule(machine_assignment, operation_sequence)
        
        # 计算目标值
        makespan, total_workload = self._calculate_objectives(schedule)
        
        return schedule, makespan, total_workload
    
    def _decode_sequence(self, sequence_part):
        """解码工序序列"""
        # 统计每个工件出现的次数
        job_counts = {}
        operation_sequence = []
        
        for job_id in sequence_part:
            if job_id not in job_counts:
                job_counts[job_id] = 0
            job_counts[job_id] += 1
            
            # 检查工序数是否有效
            if job_counts[job_id] <= len(self.problem.jobs[job_id - 1].operations):
                operation_sequence.append((job_id, job_counts[job_id]))
        
        return operation_sequence
    
    def _calculate_schedule(self, machine_assignment, operation_sequence):
        """计算调度方案"""
        # 初始化机器状态
        machine_available_time = [0] * (self.num_machines + 1)
        job_operation_finish_time = {}
        
        # 初始化工件工序完成时间
        for job in self.problem.jobs:
            job_operation_finish_time[job.id] = [0] * (len(job.operations) + 1)
        
        schedule = []
        
        # 按工序顺序处理每个工序
        for job_id, op_id in operation_sequence:
            operation = self.problem.jobs[job_id-1].operations[op_id-1]
            machine_id = machine_assignment[self._get_op_index(job_id, op_id)]
            
            # 获取该工序在选定机器上的加工时间
            processing_time = operation.machines.get(machine_id, 0)
            
            # 计算最早开始时间
            machine_time = machine_available_time[machine_id]
            
            if op_id > 1:
                prev_op_finish = job_operation_finish_time[job_id][op_id-1]
            else:
                prev_op_finish = 0
            
            start_time = max(machine_time, prev_op_finish)
            finish_time = start_time + processing_time
            
            # 更新机器可用时间
            machine_available_time[machine_id] = finish_time
            
            # 更新工件工序完成时间
            job_operation_finish_time[job_id][op_id] = finish_time
            
            # 添加到调度方案
            schedule.append((job_id, op_id, machine_id, start_time, finish_time))
        
        return schedule
    
    def _get_op_index(self, job_id, op_id):
        """获取工序在染色体中的索引"""
        index = 0
        for j in range(1, job_id):
            index += self.problem.jobs[j-1].get_operation_count()
        index += op_id - 1
        return index
    
    def _calculate_objectives(self, schedule):
        """计算目标值"""
        if not schedule:
            return float('inf'), float('inf')
            
        # 目标1: 最小化总流程时间 (makespan)
        makespan = max(op[4] for op in schedule)
        
        # 目标2: 最小化机器总负载
        machine_workload = [0] * (self.num_machines + 1)
        for job_id, op_id, machine_id, start_time, finish_time in schedule:
            operation = self.problem.jobs[job_id-1].operations[op_id-1]
            processing_time = operation.machines.get(machine_id, 0)
            machine_workload[machine_id] += processing_time
        
        total_workload = sum(machine_workload[1:])
        
        return makespan, total_workload