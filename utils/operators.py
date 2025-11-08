"""
遗传算子模块
实现交叉、变异等操作
"""

import random
import numpy as np

class GeneticOperators:
    """遗传算子类"""
    
    def __init__(self, problem):
        self.problem = problem
        self.num_jobs = problem.num_jobs
        self.num_machines = problem.machines
        self.total_operations = problem.get_total_operations()
        
        # 预计算每个工序的可用机器列表
        self.operation_machines = []
        for job in problem.jobs:
            for op in job.operations:
                self.operation_machines.append(list(op.machines.keys()))
    
    def initialize_population(self, pop_size):
        """初始化种群"""
        population = []
        for _ in range(pop_size):
            # 机器分配部分：从每个工序的可用机器中随机选择
            machine_part = []
            for i in range(self.total_operations):
                available_machines = self.operation_machines[i]
                machine_part.append(random.choice(available_machines) - 1)  # 转换为0-based
            
            # 工序排序部分：生成有效的工序序列
            sequence_part = self._generate_valid_sequence()
            
            # 合并为染色体
            chromosome = machine_part + sequence_part
            population.append(chromosome)
        
        return population
    
    def _generate_valid_sequence(self):
        """生成有效的工序序列"""
        sequence = []
        # 为每个工件生成工序序列
        for job_id in range(1, self.num_jobs + 1):
            job = self.problem.jobs[job_id - 1]
            for op_id in range(1, len(job.operations) + 1):
                sequence.append(job_id)
        
        # 随机打乱序列
        random.shuffle(sequence)
        return sequence
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        # 分离机器分配和工序排序部分
        machine_part1 = parent1[:self.total_operations]
        sequence_part1 = parent1[self.total_operations:]
        
        machine_part2 = parent2[:self.total_operations]
        sequence_part2 = parent2[self.total_operations:]
        
        # 机器分配部分：单点交叉
        if random.random() < 0.5:
            point = random.randint(1, self.total_operations - 1)
            new_machine_part1 = machine_part1[:point] + machine_part2[point:]
            new_machine_part2 = machine_part2[:point] + machine_part1[point:]
        else:
            new_machine_part1 = machine_part1[:]
            new_machine_part2 = machine_part2[:]
        
        # 工序排序部分：顺序交叉(OX)
        if random.random() < 0.5:
            new_sequence_part1 = self._order_crossover(sequence_part1, sequence_part2)
            new_sequence_part2 = self._order_crossover(sequence_part2, sequence_part1)
        else:
            new_sequence_part1 = sequence_part1[:]
            new_sequence_part2 = sequence_part2[:]
        
        # 合并为新染色体
        child1 = new_machine_part1 + new_sequence_part1
        child2 = new_machine_part2 + new_sequence_part2
        
        return child1, child2
    
    def _order_crossover(self, parent1, parent2):
        """顺序交叉操作"""
        size = len(parent1)
        child = [-1] * size
        
        # 选择两个交叉点
        point1 = random.randint(0, size - 1)
        point2 = random.randint(0, size - 1)
        start = min(point1, point2)
        end = max(point1, point2)
        
        # 复制parent1中start到end的片段到child
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # 从parent2中填充剩余位置
        parent2_idx = (end + 1) % size
        child_idx = (end + 1) % size
        
        while -1 in child:
            if parent2[parent2_idx] not in child:
                child[child_idx] = parent2[parent2_idx]
                child_idx = (child_idx + 1) % size
            parent2_idx = (parent2_idx + 1) % size
        
        return child
    
    def mutation(self, chromosome):
        """变异操作"""
        # 分离机器分配和工序排序部分
        machine_part = chromosome[:self.total_operations]
        sequence_part = chromosome[self.total_operations:]
        
        # 机器分配部分变异：从可用机器中重新选择
        if random.random() < 0.5:
            idx = random.randint(0, len(machine_part) - 1)
            available_machines = self.operation_machines[idx]
            machine_part[idx] = random.choice(available_machines) - 1
        
        # 工序排序部分变异：交换两个位置
        if random.random() < 0.5:
            idx1 = random.randint(0, len(sequence_part) - 1)
            idx2 = random.randint(0, len(sequence_part) - 1)
            sequence_part[idx1], sequence_part[idx2] = sequence_part[idx2], sequence_part[idx1]
        
        # 合并为新染色体
        return machine_part + sequence_part