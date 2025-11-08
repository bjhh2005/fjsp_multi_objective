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
    """MOPSO算法类 - 适配新的染色体结构"""
    
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
        
        # 粒子位置（使用字典结构）
        self.positions = []
        
        # 个体最优
        self.pbest_positions = []
        self.pbest_objectives = []
        
        # 全局最优(存档)
        self.archive = []
        self.archive_size = 200
        
        # 变异概率
        self.mutation_rate = 0.1
    
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
                
                # 更新粒子位置（基于离散操作）
                self._update_particle(i, leader)
                
                # 评估新位置
                schedule, makespan, total_workload = self._evaluate_individual(self.positions[i])
                objectives = [makespan, total_workload]
                
                # 更新个体最优
                if len(self.pbest_objectives[i]) == 0 or self._dominates(objectives, self.pbest_objectives[i]):
                    self.pbest_positions[i] = self._copy_chromosome(self.positions[i])
                    self.pbest_objectives[i] = objectives
                
                # 更新存档
                self._update_archive(self.positions[i], objectives)
            
            # 更新Pareto前沿
            self._update_pareto_front_from_archive()
            
            # 记录当前代
            self.generation_history.append([solution['objectives'] for solution in self.archive])
            
            # 显示进度
            if (gen + 1) % 10 == 0 or gen == self.max_gen - 1:
                print(f"Generation {gen+1}/{self.max_gen}, Archive Size: {len(self.archive)}")
        
        self.runtime = time.time() - start_time
        return self.pareto_front, self.objectives
    
    def _initialize_swarm(self):
        """初始化粒子群 - 使用新的染色体结构"""
        # 初始化位置
        self.positions = self.operators.initialize_population(self.pop_size)
        
        # 初始化个体最优
        self.pbest_positions = [self._copy_chromosome(pos) for pos in self.positions]
        self.pbest_objectives = [[] for _ in range(self.pop_size)]
        
        # 评估初始位置并更新个体最优
        for i in range(self.pop_size):
            schedule, makespan, total_workload = self._evaluate_individual(self.positions[i])
            self.pbest_objectives[i] = [makespan, total_workload]
            
            # 更新存档
            self._update_archive(self.positions[i], self.pbest_objectives[i])
    
    def _copy_chromosome(self, chromosome):
        """深度复制染色体"""
        return {
            'sequence': chromosome['sequence'][:],
            'machines': chromosome['machines'].copy()
        }
    
    def _select_leader(self) -> Optional[Dict]:
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
    
    def _calculate_crowding_distance(self, solution: Dict) -> float:
        """计算拥挤度"""
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
            crowd1 = (sorted_obj1[idx1+1] - sorted_obj1[idx1-1]) / (sorted_obj1[-1] - sorted_obj1[0] + 1e-8)
        
        if idx2 == 0 or idx2 == len(sorted_obj2) - 1:
            crowd2 = float('inf')
        else:
            crowd2 = (sorted_obj2[idx2+1] - sorted_obj2[idx2-1]) / (sorted_obj2[-1] - sorted_obj2[0] + 1e-8)
        
        return crowd1 + crowd2
    
    def _update_particle(self, i: int, leader: Optional[Dict]):
        """更新粒子位置 - 使用离散操作"""
        if leader is None:
            return
        
        # 以一定概率向个体最优学习（交叉操作）
        if random.random() < self.c1:
            child1, child2 = self.operators.crossover(self.positions[i], self.pbest_positions[i])
            self.positions[i] = child1  # 选择第一个子代
        
        # 以一定概率向全局最优学习（交叉操作）
        if random.random() < self.c2:
            child1, child2 = self.operators.crossover(self.positions[i], leader)
            self.positions[i] = child1  # 选择第一个子代
        
        # 变异操作（惯性效应）
        if random.random() < self.mutation_rate:
            self.positions[i] = self.operators.mutation(self.positions[i], self.mutation_rate)
    
    def _update_archive(self, position: Dict, objectives: List[float]):
        """更新存档"""
        # 检查是否被存档中的解支配
        dominated = False
        to_remove = []
        
        for idx, solution in enumerate(self.archive):
            if self._dominates(solution['objectives'], objectives):
                dominated = True
                break
            elif self._dominates(objectives, solution['objectives']):
                to_remove.append(idx)
        
        # 如果不被任何解支配，则加入存档
        if not dominated:
            # 移除被新解支配的解
            self.archive = [s for i, s in enumerate(self.archive) if i not in to_remove]
            
            # 添加新解
            self.archive.append({
                'position': self._copy_chromosome(position),
                'objectives': objectives[:]
            })
            
            # 如果存档超过大小限制，使用拥挤度移除一些解
            if len(self.archive) > self.archive_size:
                # 计算所有解的拥挤度
                for solution in self.archive:
                    solution['crowding_distance'] = self._calculate_crowding_distance(solution)
                
                # 按拥挤度排序，移除拥挤度小的解
                self.archive.sort(key=lambda s: s.get('crowding_distance', 0))
                self.archive = self.archive[-self.archive_size:]
    
    def _update_pareto_front_from_archive(self):
        """从存档更新Pareto前沿"""
        self.pareto_front = [solution['position'] for solution in self.archive]
        self.objectives = [solution['objectives'] for solution in self.archive]