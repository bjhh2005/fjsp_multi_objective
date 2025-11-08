"""
算法基类
定义所有多目标优化算法的通用接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

class Algorithm(ABC):
    """多目标优化算法抽象基类"""
    
    def __init__(self, problem, **kwargs):
        """
        初始化算法
        
        Args:
            problem: FJSP问题实例
            **kwargs: 算法特定参数
        """
        self.problem = problem
        self.params = kwargs
        
        # 通用属性
        self.pareto_front = []  # Pareto前沿解
        self.objectives = []    # Pareto前沿对应的目标值
        self.runtime = 0        # 算法运行时间
        self.generation_history = []  # 每代的目标值历史
        
        # 初始化解码器
        from utils.decoder import FJSPDecoder
        self.decoder = FJSPDecoder(problem)
        
        # Rich控制台
        self.console = Console()
    
    @abstractmethod
    def run(self) -> Tuple[List[Any], List[List[float]]]:
        """
        运行算法
        
        Returns:
            pareto_front: Pareto前沿解
            objectives: Pareto前沿对应的目标值
        """
        pass
    
    @abstractmethod
    def _initialize_algorithm(self):
        """初始化算法特定参数"""
        pass
    
    def get_results(self) -> dict:
        """
        获取算法结果
        
        Returns:
            包含所有结果的字典
        """
        return {
            'pareto_front': self.pareto_front,
            'objectives': self.objectives,
            'runtime': self.runtime,
            'generation_history': self.generation_history
        }
    
    def _evaluate_individual(self, chromosome):
        """
        评估个体适应度
        
        Args:
            chromosome: 染色体
            
        Returns:
            schedule: 调度方案
            makespan: 总流程时间
            total_workload: 机器总负载
        """
        return self.decoder.decode(chromosome)
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """
        判断obj1是否支配obj2
        
        Args:
            obj1: 目标值1
            obj2: 目标值2
            
        Returns:
            bool: obj1是否支配obj2
        """
        return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    
    def _update_pareto_front(self, population: List[dict]):
        """
        更新Pareto前沿
        
        Args:
            population: 种群
        """
        # 提取所有解
        all_solutions = [(ind['chromosome'], ind['objectives']) for ind in population]
        
        # 找出非支配解
        pareto_front = []
        pareto_objectives = []
        
        for i, (chromosome, objectives) in enumerate(all_solutions):
            dominated = False
            for j, (_, other_objectives) in enumerate(all_solutions):
                if i != j and self._dominates(other_objectives, objectives):
                    dominated = True
                    break
            
            if not dominated:
                pareto_front.append(chromosome)
                pareto_objectives.append(objectives)
        
        self.pareto_front = pareto_front
        self.objectives = pareto_objectives
    
    def _display_progress(self, current_gen: int, max_gen: int, pareto_size: int, 
                         best_makespan: Optional[float] = None, best_workload: Optional[float] = None):
        """
        显示当前进度
        
        Args:
            current_gen: 当前代数
            max_gen: 最大代数
            pareto_size: Pareto前沿大小
            best_makespan: 当前最优makespan
            best_workload: 当前最优workload
        """
        progress_text = f"Generation {current_gen}/{max_gen}"
        
        # 修复：处理可选参数
        if best_makespan is not None and best_workload is not None:
            info_text = f"Pareto Size: {pareto_size} | Best Makespan: {best_makespan:.1f} | Best Workload: {best_workload:.1f}"
        else:
            info_text = f"Pareto Size: {pareto_size}"
        
        self.console.print(f"[green]{progress_text}[/green] | [blue]{info_text}[/blue]")
    
    def _display_algorithm_info(self, algorithm_name: str):
        """显示算法信息"""
        info_table = Table(title=f"[bold blue]{algorithm_name} Algorithm Information[/bold blue]")
        info_table.add_column("Parameter", style="cyan")
        info_table.add_column("Value", style="magenta")
        
        info_table.add_row("Problem", f"{self.problem.num_jobs} jobs × {self.problem.machines} machines")
        info_table.add_row("Total Operations", str(self.problem.get_total_operations()))
        
        for key, value in self.params.items():
            info_table.add_row(key.replace('_', ' ').title(), str(value))
        
        self.console.print(Panel(info_table, expand=False))
    
    def _display_final_results(self, algorithm_name: str):
        """显示最终结果"""
        if not self.objectives:
            self.console.print("[red]No results found![/red]")
            return
        
        makespans = [obj[0] for obj in self.objectives]
        workloads = [obj[1] for obj in self.objectives]
        
        result_table = Table(title=f"[bold green]{algorithm_name} Final Results[/bold green]")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", style="magenta")
        
        result_table.add_row("Runtime", f"{self.runtime:.2f} seconds")
        result_table.add_row("Pareto Front Size", str(len(self.pareto_front)))
        result_table.add_row("Min Makespan", f"{min(makespans):.1f}")
        result_table.add_row("Max Makespan", f"{max(makespans):.1f}")
        result_table.add_row("Avg Makespan", f"{sum(makespans)/len(makespans):.1f}")
        result_table.add_row("Min Workload", f"{min(workloads):.1f}")
        result_table.add_row("Max Workload", f"{max(workloads):.1f}")
        result_table.add_row("Avg Workload", f"{sum(workloads)/len(workloads):.1f}")
        
        self.console.print(Panel(result_table, expand=False))