"""
主程序入口
支持通过命令行参数选择算法
"""

import argparse
import time
import pickle
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from data import FJSPProblem
from utils.decoder import FJSPDecoder
from algorithms.init import get_algorithm
from utils.visualization import plot_pareto_front, plot_gantt_chart, plot_convergence

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='柔性作业车间调度多目标优化')
    
    # 算法选择
    parser.add_argument('-a', '--algorithm', type=str, default='nsga2',
                        choices=['nsga2', 'moead', 'mopso'],
                        help='选择算法 (nsga2, moead, mopso)')
    
    # 算法参数
    parser.add_argument('--pop_size', type=int, default=100,
                        help='种群大小')
    parser.add_argument('--max_gen', type=int, default=100,
                        help='最大迭代次数')
    
    # NSGA-II特定参数
    parser.add_argument('--crossover_rate', type=float, default=0.9,
                        help='交叉概率 (NSGA-II)')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                        help='变异概率 (NSGA-II)')
    
    # MOEA/D特定参数
    parser.add_argument('--neighborhood_size', type=int, default=20,
                        help='邻域大小 (MOEA/D)')
    parser.add_argument('--delta', type=float, default=0.9,
                        help='选择邻域的概率 (MOEA/D)')
    
    # MOPSO特定参数
    parser.add_argument('--w', type=float, default=0.4,
                        help='惯性权重 (MOPSO)')
    parser.add_argument('--c1', type=float, default=2.0,
                        help='个体学习因子 (MOPSO)')
    parser.add_argument('--c2', type=float, default=2.0,
                        help='社会学习因子 (MOPSO)')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果输出目录')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出')
    
    return parser.parse_args()

def run_single_algorithm(algorithm_name, problem, args):
    """运行单个算法"""
    console = Console()
    
    console.print(f"\n[bold cyan]开始运行 {algorithm_name.upper()} 算法...[/bold cyan]")
    
    # 准备算法参数
    common_params = {
        'pop_size': args.pop_size,
        'max_gen': args.max_gen
    }
    
    if algorithm_name == 'nsga2':
        params = {
            **common_params,
            'crossover_rate': args.crossover_rate,
            'mutation_rate': args.mutation_rate
        }
    elif algorithm_name == 'moead':
        params = {
            **common_params,
            'neighborhood_size': args.neighborhood_size,
            'delta': args.delta
        }
    elif algorithm_name == 'mopso':
        params = {
            **common_params,
            'w': args.w,
            'c1': args.c1,
            'c2': args.c2
        }
    else:
        params = common_params
    
    # 获取算法类并创建实例
    AlgorithmClass = get_algorithm(algorithm_name)
    algorithm = AlgorithmClass(problem, **params)
    
    # 运行算法
    start_time = time.time()
    pareto_front, objectives = algorithm.run()
    runtime = time.time() - start_time
    
    # 获取结果
    results = algorithm.get_results()
    
    console.print(f"[green]✓ {algorithm_name.upper()} 运行完成, 耗时: {runtime:.2f}秒[/green]")
    console.print(f"[blue]✓ Pareto前沿大小: {len(pareto_front)}[/blue]")
    
    return algorithm_name, results

def main():
    """主函数"""
    console = Console()
    
    # 显示欢迎信息
    welcome_text = Text("柔性作业车间调度多目标优化系统", style="bold blue")
    console.print(Panel(welcome_text, expand=False))
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置随机种子
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        console.print(f"[yellow]随机种子设置为: {args.seed}[/yellow]")
    
    # 创建结果目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        console.print(f"[green]创建输出目录: {args.output_dir}[/green]")
    
    # 初始化问题
    console.print("\n[bold]初始化问题...[/bold]")
    problem = FJSPProblem()
    decoder = FJSPDecoder(problem)
    
    # 显示问题信息
    problem_info = f"""
    [cyan]问题规模:[/cyan]
    • 工件数量: {problem.num_jobs}
    • 机器数量: {problem.machines}
    • 总工序数: {problem.get_total_operations()}
    """
    console.print(Panel(problem_info.strip(), title="问题信息", expand=False))
    
    # 运行单个算法
    name, results = run_single_algorithm(args.algorithm, problem, args)
    
    # 保存结果
    console.print(f"\n[bold]保存结果...[/bold]")
    with open(f'{args.output_dir}/{name}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    console.print(f"[green]✓ 结果已保存到: {args.output_dir}/{name}_results.pkl[/green]")
    
    # 绘制Pareto前沿
    console.print(f"\n[bold]生成可视化结果...[/bold]")
    plot_pareto_front([results['objectives']], [name],
                        f'{args.output_dir}/{name}_pareto_front.png')
    console.print(f"[green]✓ Pareto前沿图已保存: {args.output_dir}/{name}_pareto_front.png[/green]")
    
    # 绘制收敛曲线
    plot_convergence([results['generation_history']], [name],
                    f'{args.output_dir}/{name}_convergence.png')
    console.print(f"[green]✓ 收敛曲线已保存: {args.output_dir}/{name}_convergence.png[/green]")
    
    # 绘制甘特图
    if results['pareto_front']:
        best_solution = results['pareto_front'][0]
        schedule, makespan, total_workload = decoder.decode(best_solution)
        
        console.print(f"\n[bold]绘制甘特图...[/bold]")
        console.print(f"[blue]最优解: Makespan={makespan}, Workload={total_workload}[/blue]")
        plot_gantt_chart(schedule, f'{args.output_dir}/{name}_gantt_chart.png')
        console.print(f"[green]✓ 甘特图已保存: {args.output_dir}/{name}_gantt_chart.png[/green]")

    console.print(f"\n[bold green]所有结果已保存到 '{args.output_dir}' 目录[/bold green]")

if __name__ == "__main__":
    main()