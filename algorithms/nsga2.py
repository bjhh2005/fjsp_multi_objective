"""
NSGA-II算法实现
继承自Algorithm基类
"""

import random
import time
from typing import List, Tuple, Any, Dict, Union
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from algorithms.base import Algorithm
from utils.operators import GeneticOperators

class NSGA2(Algorithm):
    """NSGA-II算法类"""
    
    def __init__(self, problem, pop_size=100, max_gen=100, crossover_rate=0.9, mutation_rate=0.1):
        """
        初始化NSGA-II算法
        
        Args:
            problem: FJSP问题实例
            pop_size: 种群大小
            max_gen: 最大迭代次数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
        """
        super().__init__(problem, pop_size=pop_size, max_gen=max_gen, 
                        crossover_rate=crossover_rate, mutation_rate=mutation_rate)
        
        # 算法特定参数
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # 初始化算法
        self._initialize_algorithm()
    
    def _initialize_algorithm(self):
        """初始化NSGA-II算法特定参数"""
        self.operators = GeneticOperators(self.problem)
    
    def run(self) -> Tuple[List[Any], List[List[float]]]:
        """运行NSGA-II算法"""
        # 显示算法信息
        self._display_algorithm_info("NSGA-II")
        
        start_time = time.time()
        
        # 初始化种群
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Initializing population...", total=None)
            population = self.operators.initialize_population(self.pop_size)
            progress.update(task, description="Population initialized")
            
            # 评估初始种群
            progress.update(task, description="Evaluating initial population...")
            evaluated_pop = self._evaluate_population(population)
            
            # 记录初始代
            self.generation_history.append([ind['objectives'] for ind in evaluated_pop])
            
            # 主循环
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as gen_progress:
                gen_task = gen_progress.add_task("Evolving generations...", total=self.max_gen)
                
                for gen in range(self.max_gen):
                    # 选择
                    mating_pool = self._selection(evaluated_pop)
                    
                    # 交叉和变异
                    offspring = []
                    for i in range(0, len(mating_pool), 2):
                        if i+1 < len(mating_pool) and random.random() < self.crossover_rate:
                            child1, child2 = self.operators.crossover(mating_pool[i], mating_pool[i+1])
                            offspring.append(child1)
                            offspring.append(child2)
                        else:
                            offspring.append(mating_pool[i])
                            if i+1 < len(mating_pool):
                                offspring.append(mating_pool[i+1])
                    
                    # 变异
                    for i in range(len(offspring)):
                        if random.random() < self.mutation_rate:
                            offspring[i] = self.operators.mutation(offspring[i])
                    
                    # 评估子代
                    evaluated_offspring = self._evaluate_population(offspring)
                    
                    # 环境选择
                    evaluated_pop = self._environmental_selection(evaluated_pop, evaluated_offspring)
                    
                    # 更新Pareto前沿
                    self._update_pareto_front(evaluated_pop)
                    
                    # 记录当前代
                    self.generation_history.append([ind['objectives'] for ind in evaluated_pop])
                    
                    # 更新进度
                    best_makespan = min(obj[0] for obj in self.objectives) if self.objectives else None
                    best_workload = min(obj[1] for obj in self.objectives) if self.objectives else None
                    
                    gen_progress.update(gen_task, advance=1, 
                                     description=f"Gen {gen+1}/{self.max_gen} | Pareto: {len(self.pareto_front)}")
                    
                    # 显示详细进度（每10代或最后一代）
                    if (gen + 1) % 10 == 0 or gen == self.max_gen - 1:
                        self._display_progress(gen + 1, self.max_gen, len(self.pareto_front), 
                                            best_makespan, best_workload)
        
        self.runtime = time.time() - start_time
        
        # 显示最终结果
        self._display_final_results("NSGA-II")
        
        return self.pareto_front, self.objectives
    
    def _evaluate_population(self, population: List) -> List[dict]:
        """评估种群"""
        evaluated_pop = []
        for chromosome in population:
            schedule, makespan, total_workload = self._evaluate_individual(chromosome)
            evaluated_pop.append({
                'chromosome': chromosome,
                'objectives': [makespan, total_workload],
                'rank': 0,
                'crowding_distance': 0
            })
        return evaluated_pop
    
    def _fast_non_dominated_sort(self, population: List[dict]) -> List[List[int]]:
        """快速非支配排序"""
        fronts = [[]]
        
        for i, individual in enumerate(population):
            individual['domination_count'] = 0
            individual['dominated_set'] = []
            
            for j, other in enumerate(population):
                if i != j:
                    if self._dominates(individual['objectives'], other['objectives']):
                        individual['dominated_set'].append(j)
                    elif self._dominates(other['objectives'], individual['objectives']):
                        individual['domination_count'] += 1
            
            if individual['domination_count'] == 0:
                individual['rank'] = 0
                fronts[0].append(i)
        
        # 构建后续前沿
        k = 0
        while len(fronts[k]) > 0:
            next_front = []
            for i in fronts[k]:
                for j in population[i]['dominated_set']:
                    population[j]['domination_count'] -= 1
                    if population[j]['domination_count'] == 0:
                        population[j]['rank'] = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # 移除最后一个空前沿
    
    def _calculate_crowding_distance(self, population: List[dict], front: List[int]):
        """计算拥挤距离"""
        if len(front) == 0:
            return
        
        # 初始化拥挤距离
        for i in front:
            population[i]['crowding_distance'] = 0
        
        # 每个目标函数单独计算
        num_objectives = len(population[front[0]]['objectives'])
        for m in range(num_objectives):
            # 按第m个目标函数值排序
            sorted_front = sorted(front, key=lambda i: population[i]['objectives'][m])
            
            # 边界点设置为无穷大
            population[sorted_front[0]]['crowding_distance'] = float('inf')
            population[sorted_front[-1]]['crowding_distance'] = float('inf')
            
            # 计算中间点的拥挤距离
            obj_min = population[sorted_front[0]]['objectives'][m]
            obj_max = population[sorted_front[-1]]['objectives'][m]
            
            if obj_max - obj_min == 0:
                continue
                
            for i in range(1, len(sorted_front) - 1):
                distance = (population[sorted_front[i+1]]['objectives'][m] - 
                           population[sorted_front[i-1]]['objectives'][m]) / (obj_max - obj_min)
                population[sorted_front[i]]['crowding_distance'] += distance
    
    def _selection(self, population: List[dict]) -> List:
        """选择操作"""
        # 锦标赛选择
        mating_pool = []
        for _ in range(len(population)):
            # 随机选择两个个体
            i, j = random.sample(range(len(population)), 2)
            
            # 比较rank和crowding_distance
            if population[i]['rank'] < population[j]['rank']:
                mating_pool.append(population[i]['chromosome'])
            elif population[i]['rank'] > population[j]['rank']:
                mating_pool.append(population[j]['chromosome'])
            else:
                # rank相同，选择拥挤距离大的
                if population[i]['crowding_distance'] > population[j]['crowding_distance']:
                    mating_pool.append(population[i]['chromosome'])
                else:
                    mating_pool.append(population[j]['chromosome'])
        
        return mating_pool
    
    def _environmental_selection(self, pop: List[dict], offspring: List[dict]) -> List[dict]:
        """环境选择"""
        combined_pop = pop + offspring
        
        # 快速非支配排序
        fronts = self._fast_non_dominated_sort(combined_pop)
        
        # 计算拥挤距离
        for front in fronts:
            self._calculate_crowding_distance(combined_pop, front)
        
        # 选择新种群
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                for i in front:
                    new_pop.append(combined_pop[i])
            else:
                # 按拥挤距离排序，选择拥挤距离大的
                remaining = self.pop_size - len(new_pop)
                sorted_front = sorted(front, key=lambda i: combined_pop[i]['crowding_distance'], reverse=True)
                for i in range(remaining):
                    new_pop.append(combined_pop[sorted_front[i]])
                break
        
        return new_pop