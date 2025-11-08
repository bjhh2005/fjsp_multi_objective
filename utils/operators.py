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
        """基于工件的编码"""
        population = []
        for _ in range(pop_size):
            # 工序序列：使用工件ID重复出现表示工序
            sequence_part = []
            for job_id, job in enumerate(self.problem.jobs):
                sequence_part.extend([job_id] * len(job.operations))
            random.shuffle(sequence_part)
            
            # 机器分配：直接使用字典或列表保持对应关系
            machine_assignment = {}
            for job_id, job in enumerate(self.problem.jobs):
                for op_id, op in enumerate(job.operations):
                    available_machines = list(op.machines.keys())
                    machine_assignment[(job_id, op_id)] = random.choice(available_machines) - 1
            
            chromosome = {
                'sequence': sequence_part,
                'machines': machine_assignment
            }
            population.append(chromosome)
        
        return population
    
    def _generate_valid_sequence(self):
        """生成有效的工序序列"""
        sequence = []
        # 为每个工件生成工序序列
        for job_id in range(1, self.num_jobs + 1):
            job = self.problem.jobs[job_id - 1]
            sequence.extend(list([job_id]) * len(job.operations))
        
        # 随机打乱序列
        random.shuffle(sequence)
        return sequence
    
    def crossover(self, parent1, parent2):
        """POX交叉操作，与文章描述一致"""
        # 分离染色体部分
        machine_dict1 = parent1['machines']
        sequence1 = parent1['sequence']
        machine_dict2 = parent2['machines'] 
        sequence2 = parent2['sequence']
        
        # 随机划分工件集合为两个非空子集
        all_jobs = list(range(self.num_jobs))
        random.shuffle(all_jobs)
        split_point = random.randint(1, len(all_jobs) - 1)
        J1 = set(all_jobs[:split_point])  # 子集J1
        # J2 = set(all_jobs[split_point:])  
        # 子集J2
        
        # 工序序列POX交叉
        child_sequence1 = self._pox_crossover(sequence1, sequence2, J1)
        child_sequence2 = self._pox_crossover(sequence2, sequence1, J1)
        
        # 机器分配均匀交叉
        child_machines1 = {}
        child_machines2 = {}
        
        for job_id in range(self.num_jobs):
            job = self.problem.jobs[job_id]
            for op_id in range(len(job.operations)):
                # 以概率0.5选择父代的机器分配
                if random.random() < 0.5:
                    child_machines1[(job_id, op_id)] = machine_dict1[(job_id, op_id)]
                    child_machines2[(job_id, op_id)] = machine_dict2[(job_id, op_id)]
                else:
                    child_machines1[(job_id, op_id)] = machine_dict2[(job_id, op_id)]
                    child_machines2[(job_id, op_id)] = machine_dict1[(job_id, op_id)]
        
        # 构建子代染色体
        child1 = {
            'sequence': child_sequence1,
            'machines': child_machines1
        }
        child2 = {
            'sequence': child_sequence2, 
            'machines': child_machines2
        }
        
        return child1, child2

    def _pox_crossover(self, parent1, parent2, J1):
        """POX交叉的具体实现"""
        child = [None] * len(parent1)
        
        # 从parent1复制属于J1的工件工序
        J1_positions = []
        for i, job_id in enumerate(parent1):
            if job_id in J1:
                child[i] = job_id
                J1_positions.append(i)
        
        # 从parent2填充属于J2的工件工序（保持相对顺序）
        parent2_ptr = 0
        for i in range(len(child)):
            if child[i] is None:
                # 找到parent2中下一个属于J2的工件
                while parent2[parent2_ptr] in J1:
                    parent2_ptr += 1
                child[i] = parent2[parent2_ptr]
                parent2_ptr += 1
        
        return child
    
    '''
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
    '''
    
    def mutation(self, chromosome, p_m=0.1):
        """变异操作 - 精确对应数学描述"""
        mutated_chromosome = {
            'sequence': chromosome['sequence'][:],
            'machines': chromosome['machines'].copy()
        }
        
        L = len(mutated_chromosome['sequence'])
        
        # 工序交换变异：随机选择两个位置i和j交换工序
        if random.random() < p_m:
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
            # 数学表示：π' = (π₁, ..., πⱼ, ..., πᵢ, ..., π_L)
            pi = mutated_chromosome['sequence']
            pi[i], pi[j] = pi[j], pi[i]
        
        # 机器重选变异：以概率p_m重新选择机器
        for job_id in range(self.num_jobs):
            job = self.problem.jobs[job_id]
            for op_id in range(len(job.operations)):
                if random.random() < p_m:
                    operation = job.operations[op_id]
                    available_machines = list(operation.machines.keys())
                    new_machine = random.choice(available_machines) - 1
                    # 满足约束：m_ji ∈ M_ji
                    mutated_chromosome['machines'][(job_id, op_id)] = new_machine
        
        return mutated_chromosome