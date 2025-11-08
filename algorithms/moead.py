"""
MOEA/D算法实现
继承自Algorithm基类
"""

import random
import time
import numpy as np
from typing import List, Tuple, Any
from algorithms.base import Algorithm
from utils.operators import GeneticOperators

class MOEAD(Algorithm):
    """MOEA/D算法类"""
    
    def __init__(self, problem, pop_size=100, max_gen=100, neighborhood_size=20, delta=0.9):
        """
        初始化MOEA/D算法
        
        Args:
            problem: FJSP问题实例
            pop_size: 种群大小
            max_gen: 最大迭代次数
            neighborhood_size: 邻域大小
            delta: 选择邻域的概率
        """
        super().__init__(problem, pop_size=pop_size, max_gen=max_gen, 
                        neighborhood_size=neighborhood_size, delta=delta)
        
        # 算法特定参数
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.neighborhood_size = neighborhood_size
        self.delta = delta
        
        # 初始化算法
        self._initialize_algorithm()
    
    def _initialize_algorithm(self):
        """初始化MOEA/D算法特定参数"""
        self.operators = GeneticOperators(self.problem)
        
        # 权重向量
        self.weights = self._generate_weights()
        
        # 邻域
        self.neighborhoods = self._calculate_neighborhoods()
        
        # 参考点
        self.ideal_point = [float('inf'), float('inf')]
        self.nadir_point = [0, 0]
    
    def _generate_weights(self) -> List[List[float]]:
        """生成权重向量"""
        weights = []
        h = 1.0  # 精度
        
        for i in range(self.pop_size):
            w1 = i * h
            w2 = 1.0 - w1
            weights.append([w1, w2])
        
        return weights
    
    def _calculate_neighborhoods(self) -> List[List[int]]:
        """计算邻域"""
        neighborhoods = []
        for i in range(self.pop_size):
            # 计算与其他权重向量的距离
            distances = []
            for j in range(self.pop_size):
                dist = np.linalg.norm(np.array(self.weights[i]) - np.array(self.weights[j]))
                distances.append((dist, j))
            
            # 按距离排序，选择前neighborhood_size个作为邻域
            distances.sort()
            neighborhood = [j for _, j in distances[:self.neighborhood_size]]
            neighborhoods.append(neighborhood)
        
        return neighborhoods
    
    def run(self) -> Tuple[List[Any], List[List[float]]]:
        """运行MOEA/D算法"""
        start_time = time.time()
        
        # 初始化种群
        population = self.operators.initialize_population(self.pop_size)
        
        # 评估初始种群
        evaluated_pop = self._evaluate_population(population)
        
        # 更新参考点
        self._update_reference_point(evaluated_pop)
        
        # 记录初始代
        self.generation_history.append([ind['objectives'] for ind in evaluated_pop])
        
        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                # 选择邻域或整个种群
                if random.random() < self.delta:
                    mating_pool_indices = self.neighborhoods[i]
                else:
                    mating_pool_indices = list(range(self.pop_size))
                
                # 从邻域中选择两个父代
                idx1, idx2 = random.sample(mating_pool_indices, 2)
                parent1 = evaluated_pop[idx1]['chromosome']
                parent2 = evaluated_pop[idx2]['chromosome']
                
                # 交叉和变异
                child1, child2 = self.operators.crossover(parent1, parent2)
                if random.random() < 0.5:
                    child = self.operators.mutation(child1)
                else:
                    child = self.operators.mutation(child2)
                
                # 评估子代
                schedule, makespan, total_workload = self._evaluate_individual(child)
                child_objectives = [makespan, total_workload]
                
                # 更新参考点
                self._update_reference_point_with_objectives(child_objectives)
                
                # 更新邻域解
                for j in mating_pool_indices:
                    weight = self.weights[j]
                    current_obj = evaluated_pop[j]['objectives']
                    
                    # 计算加权切比雪夫距离
                    current_value = self._weighted_tchebycheff(current_obj, weight)
                    child_value = self._weighted_tchebycheff(child_objectives, weight)
                    
                    if child_value < current_value:
                        # 修复：确保正确更新字典项
                        evaluated_pop[j] = {
                            'chromosome': child,
                            'objectives': child_objectives
                        }
            
            # 更新Pareto前沿
            self._update_pareto_front(evaluated_pop)
            
            # 记录当前代
            self.generation_history.append([ind['objectives'] for ind in evaluated_pop])
            
            print(f"Generation {gen+1}/{self.max_gen}, Pareto Front Size: {len(self.pareto_front)}")
        
        self.runtime = time.time() - start_time
        return self.pareto_front, self.objectives
    
    def _evaluate_population(self, population: List) -> List[dict]:
        """评估种群"""
        evaluated_pop = []
        for chromosome in population:
            schedule, makespan, total_workload = self._evaluate_individual(chromosome)
            evaluated_pop.append({
                'chromosome': chromosome,
                'objectives': [makespan, total_workload]
            })
        return evaluated_pop
    
    def _update_reference_point(self, population: List[dict]):
        """更新参考点"""
        for individual in population:
            for i in range(2):
                if individual['objectives'][i] < self.ideal_point[i]:
                    self.ideal_point[i] = individual['objectives'][i]
                if individual['objectives'][i] > self.nadir_point[i]:
                    self.nadir_point[i] = individual['objectives'][i]
    
    def _update_reference_point_with_objectives(self, objectives: List[float]):
        """用目标值更新参考点"""
        for i in range(2):
            if objectives[i] < self.ideal_point[i]:
                self.ideal_point[i] = objectives[i]
            if objectives[i] > self.nadir_point[i]:
                self.nadir_point[i] = objectives[i]
    
    def _weighted_tchebycheff(self, objectives: List[float], weights: List[float]) -> float:
        """计算加权切比雪夫距离"""
        max_diff = 0
        for i in range(2):
            diff = abs(objectives[i] - self.ideal_point[i])
            if self.nadir_point[i] - self.ideal_point[i] > 0:
                diff = diff / (self.nadir_point[i] - self.ideal_point[i])
            diff = diff * weights[i]
            if diff > max_diff:
                max_diff = diff
        return max_diff