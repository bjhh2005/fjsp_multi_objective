"""
MOPSO算法实现
继承自Algorithm基类
"""

import random
import time
import numpy as np
from typing import List, Tuple, Any, Dict, Union, Optional
from algorithms.base import Algorithm
from utils.operators import GeneticOperators

class MOPSO(Algorithm):
    """MOPSO算法类"""
    
    def __init__(self, problem, pop_size=100, max_gen=100, w=0.4, c1=2.0, c2=2.0):
        """
        初始化MOPSO算法
        
        Args:
            problem: FJSP问题实例
            pop_size: 种群大小
            max_gen: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
        """
        super().__init__(problem, pop_size=pop_size, max_gen=max_gen, 
                        w=w, c1=c1, c2=c2)
        
        # 算法特定参数
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # 初始化算法
        self._initialize_algorithm()
    
    def _initialize_algorithm(self):
        """初始化MOPSO算法特定参数"""
        self.operators = GeneticOperators(self.problem)
        
        # 粒子位置和速度
        self.positions = []
        self.velocities = []
        
        # 个体最优
        self.pbest_positions = []
        self.pbest_objectives = []
        
        # 全局最优(存档)
        self.archive = []
        self.archive_size = 100
    
    def run(self) -> Tuple[List[Any], List[List[float]]]:
        """运行MOPSO算法"""
        start_time = time.time()
        
        # 初始化粒子群
        self._initialize_swarm()
        
        # 记录初始代
        self.generation_history.append(self.pbest_objectives)
        
        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                # 选择全局最优领导者
                leader = self._select_leader()
                
                # 修复：只有当leader不为None时才更新速度
                if leader is not None:
                    self._update_velocity(i, leader)
                    self._update_position(i)
                
                # 评估新位置
                schedule, makespan, total_workload = self._evaluate_individual(self.positions[i])
                objectives = [makespan, total_workload]
                
                # 更新个体最优
                if len(self.pbest_objectives[i]) == 0 or self._dominates(objectives, self.pbest_objectives[i]):
                    self.pbest_positions[i] = self.positions[i][:]
                    self.pbest_objectives[i] = objectives
                
                # 更新存档
                self._update_archive(self.positions[i], objectives)
            
            # 更新Pareto前沿
            self._update_pareto_front_from_archive()
            
            # 记录当前代
            self.generation_history.append([solution['objectives'] for solution in self.archive])
            
            print(f"Generation {gen+1}/{self.max_gen}, Archive Size: {len(self.archive)}")
        
        self.runtime = time.time() - start_time
        return self.pareto_front, self.objectives
    
    def _initialize_swarm(self):
        """初始化粒子群"""
        # 初始化位置
        self.positions = self.operators.initialize_population(self.pop_size)
        
        # 初始化速度
        total_length = len(self.positions[0])
        for _ in range(self.pop_size):
            velocity = []
            for _ in range(total_length):
                # 速度范围为[-1, 1]
                velocity.append(random.uniform(-1, 1))
            self.velocities.append(velocity)
        
        # 初始化个体最优
        self.pbest_positions = [pos[:] for pos in self.positions]
        self.pbest_objectives = [[] for _ in range(self.pop_size)]
        
        # 评估初始位置并更新个体最优
        for i in range(self.pop_size):
            schedule, makespan, total_workload = self._evaluate_individual(self.positions[i])
            self.pbest_objectives[i] = [makespan, total_workload]
            
            # 更新存档
            self._update_archive(self.positions[i], self.pbest_objectives[i])
    
    def _select_leader(self) -> Optional[List]:
        """选择全局最优领导者"""
        if len(self.archive) == 0:
            return None
        
        # 使用锦标赛选择
        k = min(5, len(self.archive))  # 锦标赛大小
        candidates = random.sample(self.archive, k)
        
        # 选择拥挤度最大的个体
        best = candidates[0]
        for candidate in candidates[1:]:
            if self._calculate_crowding_distance(candidate) > self._calculate_crowding_distance(best):
                best = candidate
        
        return best['position']
    
    def _calculate_crowding_distance(self, solution: Dict[str, Union[List, List[float]]]) -> float:
        """计算拥挤度"""
        # 简化版拥挤度计算
        if len(self.archive) <= 2:
            return float('inf')
        
        # 找到相同目标值的解
        same_obj_solutions = [s for s in self.archive if s['objectives'] == solution['objectives']]
        if len(same_obj_solutions) > 1:
            return 0  # 有重复解，拥挤度为0
        
        # 计算在目标空间中的拥挤度
        objectives = [s['objectives'] for s in self.archive]
        obj1_values = [obj[0] for obj in objectives]
        obj2_values = [obj[1] for obj in objectives]
        
        # 找到解在排序中的位置
        sorted_obj1 = sorted(obj1_values)
        sorted_obj2 = sorted(obj2_values)
        
        idx1 = sorted_obj1.index(solution['objectives'][0])
        idx2 = sorted_obj2.index(solution['objectives'][1])
        
        # 计算拥挤度
        if idx1 == 0 or idx1 == len(sorted_obj1) - 1:
            crowd1 = float('inf')
        else:
            crowd1 = (sorted_obj1[idx1+1] - sorted_obj1[idx1-1]) / (sorted_obj1[-1] - sorted_obj1[0])
        
        if idx2 == 0 or idx2 == len(sorted_obj2) - 1:
            crowd2 = float('inf')
        else:
            crowd2 = (sorted_obj2[idx2+1] - sorted_obj2[idx2-1]) / (sorted_obj2[-1] - sorted_obj2[0])
        
        return crowd1 + crowd2
    
    def _update_velocity(self, i: int, leader: List):
        """更新粒子速度"""
        for j in range(len(self.velocities[i])):
            # 机器分配部分使用连续值
            if j < self.problem.get_total_operations():
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (self.pbest_positions[i][j] - self.positions[i][j])
                social = self.c2 * r2 * (leader[j] - self.positions[i][j])
                self.velocities[i][j] = self.w * self.velocities[i][j] + cognitive + social
                
                # 限制速度范围
                self.velocities[i][j] = max(-5, min(5, self.velocities[i][j]))
            # 工序排序部分使用离散值
            else:
                # 简化处理，工序排序部分的速度不更新
                pass
    
    def _update_position(self, i: int):
        """更新粒子位置"""
        for j in range(len(self.positions[i])):
            # 机器分配部分使用连续值
            if j < self.problem.get_total_operations():
                self.positions[i][j] += self.velocities[i][j]
                # 确保在有效范围内
                self.positions[i][j] = max(0, min(self.problem.machines - 1, self.positions[i][j]))
            # 工序排序部分使用离散值
            else:
                # 随机交换两个位置
                if random.random() < 0.1:  # 10%的概率进行交换
                    idx1 = random.randint(self.problem.get_total_operations(), len(self.positions[i]) - 1)
                    idx2 = random.randint(self.problem.get_total_operations(), len(self.positions[i]) - 1)
                    self.positions[i][idx1], self.positions[i][idx2] = self.positions[i][idx2], self.positions[i][idx1]
    
    def _update_archive(self, position: List, objectives: List[float]):
        """更新存档"""
        # 检查是否被存档中的解支配
        dominated = False
        for solution in self.archive:
            if self._dominates(solution['objectives'], objectives):
                dominated = True
                break
        
        # 如果不被任何解支配，则加入存档
        if not dominated:
            # 移除被新解支配的解
            self.archive = [s for s in self.archive if not self._dominates(objectives, s['objectives'])]
            
            # 添加新解
            self.archive.append({
                'position': position[:],
                'objectives': objectives[:]
            })
            
            # 如果存档超过大小限制，使用拥挤度移除一些解
            if len(self.archive) > self.archive_size:
                self.archive.sort(key=lambda s: self._calculate_crowding_distance(s))
                self.archive = self.archive[-self.archive_size:]
    
    def _update_pareto_front_from_archive(self):
        """从存档更新Pareto前沿"""
        self.pareto_front = [solution['position'] for solution in self.archive]
        self.objectives = [solution['objectives'] for solution in self.archive]