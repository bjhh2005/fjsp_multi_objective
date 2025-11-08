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
    
    def __init__(self, problem, pop_size=100, max_gen=100, neighborhood_size=20, delta=0.9, 
                 crossover_rate=0.9, mutation_rate=0.2):
        """
        初始化MOEA/D算法
        
        Args:
            problem: FJSP问题实例
            pop_size: 种群大小
            max_gen: 最大迭代次数
            neighborhood_size: 邻域大小
            delta: 选择邻域的概率
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
        """
        super().__init__(problem, pop_size=pop_size, max_gen=max_gen, 
                        neighborhood_size=neighborhood_size, delta=delta,
                        crossover_rate=crossover_rate, mutation_rate=mutation_rate)
        
        # 算法特定参数
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.neighborhood_size = neighborhood_size
        self.delta = delta
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # 初始化算法
        self._initialize_algorithm()
    
    def _initialize_algorithm(self):
        """初始化MOEA/D算法特定参数"""
        self.operators = GeneticOperators(self.problem)
        
        # 权重向量 - 修复：使用更合理的权重生成策略
        self.weights = self._generate_weights()
        
        # 邻域
        self.neighborhoods = self._calculate_neighborhoods()
        
        # 参考点
        self.ideal_point = [float('inf'), float('inf')]
        
        # 修复：移除nadir_point，在MOEA/D中通常不需要
    
    def _generate_weights(self) -> List[List[float]]:
        """生成权重向量 - 修复：使用均匀分布的权重生成方法"""
        weights = []
        
        # 对于2目标问题，使用均匀分布的权重
        if self.pop_size <= 1:
            weights = [[0.5, 0.5]]
        else:
            for i in range(self.pop_size):
                w1 = i / (self.pop_size - 1) if self.pop_size > 1 else 0.5
                w2 = 1.0 - w1
                weights.append([w1, w2])
        
        return weights
    
    def _calculate_neighborhoods(self) -> List[List[int]]:
        """计算邻域 - 修复：确保邻域大小不超过种群大小"""
        neighborhoods = []
        for i in range(self.pop_size):
            # 计算与其他权重向量的距离
            distances = []
            for j in range(self.pop_size):
                if i != j:  # 排除自身
                    dist = np.linalg.norm(np.array(self.weights[i]) - np.array(self.weights[j]))
                    distances.append((dist, j))
            
            # 按距离排序，选择前neighborhood_size个作为邻域
            distances.sort()
            actual_neighborhood_size = min(self.neighborhood_size, len(distances))
            neighborhood = [j for _, j in distances[:actual_neighborhood_size]]
            neighborhoods.append(neighborhood)
        
        return neighborhoods
    
    def _weighted_tchebycheff(self, objectives: List[float], weights: List[float]) -> float:
        """计算加权切比雪夫距离 - 修复：正确的切比雪夫距离计算"""
        max_value = 0
        for i in range(2):
            # 切比雪夫距离：max(w_i * |f_i - z_i|)
            weighted_diff = weights[i] * abs(objectives[i] - self.ideal_point[i])
            if weighted_diff > max_value:
                max_value = weighted_diff
        return max_value
    
    def _update_solutions(self, child_chromosome, child_objectives, evaluated_pop):
        """更新解 - 修复：分离更新逻辑"""
        updated_count = 0
        
        for i in range(self.pop_size):
            # 计算子代在当前权重下的适应度值
            child_fitness = self._weighted_tchebycheff(child_objectives, self.weights[i])
            current_fitness = self._weighted_tchebycheff(evaluated_pop[i]['objectives'], self.weights[i])
            
            # 如果子代更好，则替换
            if child_fitness < current_fitness:
                evaluated_pop[i] = {
                    'chromosome': child_chromosome.copy() if hasattr(child_chromosome, 'copy') else child_chromosome,
                    'objectives': child_objectives.copy()
                }
                updated_count += 1
        
        return updated_count
    
    def run(self) -> Tuple[List[Any], List[List[float]]]:
        """运行MOEA/D算法 - 修复：改进算法流程"""
        start_time = time.time()
        
        # 初始化种群
        population = self.operators.initialize_population(self.pop_size)
        
        # 评估初始种群
        evaluated_pop = self._evaluate_population(population)
        
        # 更新参考点
        self._update_reference_point(evaluated_pop)
        
        # 记录初始代
        self.generation_history.append([ind['objectives'] for ind in evaluated_pop])
        
        print(f"MOEA/D开始运行，种群大小: {self.pop_size}, 最大代数: {self.max_gen}")
        
        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                # 选择邻域或整个种群
                if random.random() < self.delta and self.neighborhoods[i]:
                    mating_pool_indices = self.neighborhoods[i]
                else:
                    mating_pool_indices = list(range(self.pop_size))
                
                # 从交配池中选择两个不同的父代
                if len(mating_pool_indices) < 2:
                    # 如果交配池太小，从整个种群选择
                    idx1, idx2 = random.sample(range(self.pop_size), 2)
                else:
                    idx1, idx2 = random.sample(mating_pool_indices, 2)
                
                parent1 = evaluated_pop[idx1]['chromosome']
                parent2 = evaluated_pop[idx2]['chromosome']
                
                # 交叉操作
                if random.random() < self.crossover_rate:
                    child1, child2 = self.operators.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # 选择其中一个子代进行变异
                child = child1 if random.random() < 0.5 else child2
                
                # 变异操作
                if random.random() < self.mutation_rate:
                    child = self.operators.mutation(child)
                
                # 评估子代
                try:
                    schedule, makespan, total_workload = self._evaluate_individual(child)
                    child_objectives = [makespan, total_workload]
                    
                    # 更新参考点
                    self._update_reference_point_with_objectives(child_objectives)
                    
                    # 更新解
                    self._update_solutions(child, child_objectives, evaluated_pop)
                    
                except Exception as e:
                    print(f"评估子代时出错: {e}")
                    continue
            
            # 更新Pareto前沿
            self._update_pareto_front(evaluated_pop)
            
            # 记录当前代
            self.generation_history.append([ind['objectives'] for ind in evaluated_pop])
            
            if (gen + 1) % 50 == 0:
                print(f"Generation {gen+1}/{self.max_gen}, Pareto Front Size: {len(self.pareto_front)}")
        
        self.runtime = time.time() - start_time
        
        # 提取目标值
        self.objectives = [ind['objectives'] for ind in evaluated_pop]
        
        print(f"MOEA/D运行完成，耗时: {self.runtime:.2f}秒")
        print(f"最终Pareto前沿大小: {len(self.pareto_front)}")
        
        return self.pareto_front, self.objectives
    
    def _evaluate_population(self, population: List) -> List[dict]:
        """评估种群"""
        evaluated_pop = []
        for chromosome in population:
            try:
                schedule, makespan, total_workload = self._evaluate_individual(chromosome)
                evaluated_pop.append({
                    'chromosome': chromosome,
                    'objectives': [makespan, total_workload]
                })
            except Exception as e:
                print(f"评估染色体时出错: {e}")
                # 使用一个较差的解作为默认值
                evaluated_pop.append({
                    'chromosome': chromosome,
                    'objectives': [float('inf'), float('inf')]
                })
        return evaluated_pop
    
    def _update_reference_point(self, population: List[dict]):
        """更新参考点"""
        for individual in population:
            for i in range(2):
                if individual['objectives'][i] < self.ideal_point[i]:
                    self.ideal_point[i] = individual['objectives'][i]
    
    def _update_reference_point_with_objectives(self, objectives: List[float]):
        """用目标值更新参考点"""
        for i in range(2):
            if objectives[i] < self.ideal_point[i]:
                self.ideal_point[i] = objectives[i]
    
    def get_results(self):
        """获取算法结果 - 修复：确保返回完整且一致的结果字典"""
        # 确保我们有有效的目标值
        if not hasattr(self, 'objectives') or not self.objectives:
            # 从当前种群生成目标值
            if hasattr(self, 'evaluated_pop') and self.evaluated_pop:
                self.objectives = [ind['objectives'] for ind in self.evaluated_pop]
            else:
                self.objectives = []
        
        # 确保 Pareto 前沿和目标值数量一致
        if len(self.pareto_front) != len(self.objectives):
            print(f"MOEA/D 警告: Pareto前沿({len(self.pareto_front)})和目标值({len(self.objectives)})数量不一致")
            # 调整到最小长度
            min_len = min(len(self.pareto_front), len(self.objectives))
            if min_len > 0:
                self.pareto_front = self.pareto_front[:min_len]
                self.objectives = self.objectives[:min_len]
            else:
                # 如果都没有有效解，创建空列表
                self.pareto_front = []
                self.objectives = []
        
        return {
            'population': getattr(self, 'population', []),
            'pareto_front': self.pareto_front,
            'objectives': self.objectives,
            'generation_history': self.generation_history,
            'runtime': self.runtime,
            'algorithm_params': {
                'pop_size': self.pop_size,
                'max_gen': self.max_gen,
                'neighborhood_size': self.neighborhood_size,
                'delta': self.delta,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate
            }
        }