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
import numpy as np
import csv

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

def run_single_algorithm(algorithm_name, problem, args, console):
    """运行单个算法"""
    
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
    
    try:
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
        
        return algorithm_name, results, None
        
    except Exception as e:
        console.print(f"[red]✗ {algorithm_name.upper()} 运行失败: {str(e)}[/red]")
        return algorithm_name, None, str(e)

def main():
    """主函数"""
    console = Console()
    
    try:
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
        name, results, error = run_single_algorithm(args.algorithm, problem, args, console)
        
        if error:
            console.print(f"\n[bold red]算法执行失败: {error}[/bold red]")
            return 1
        
        if not results or not results.get('pareto_front'):
            console.print(f"\n[bold red]未获得有效结果[/bold red]")
            return 1
        
        console.print(f"\n[bold]保存结果...[/bold]")
        try:
            # 保存pickle格式（二进制，用于程序读取）
            with open(f'{args.output_dir}/{name}_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            console.print(f"[green]✓ 结果已保存到: {args.output_dir}/{name}_results.pkl[/green]")
            
            # 新增：保存文本格式（可读，用于分析）
            save_text_results(results, f'{args.output_dir}/{name}_results.txt')
            console.print(f"[green]✓ 文本结果已保存到: {args.output_dir}/{name}_results.txt[/green]")
            
            # 新增：保存CSV格式（用于Excel分析）
            save_csv_results(results, f'{args.output_dir}/{name}_pareto_front.csv')
            console.print(f"[green]✓ CSV结果已保存到: {args.output_dir}/{name}_pareto_front.csv[/green]")
            
            # 新增：保存最优解的详细信息
            save_best_solution_details(results, decoder, f'{args.output_dir}/{name}_best_solution.txt')
            console.print(f"[green]✓ 最优解详情已保存到: {args.output_dir}/{name}_best_solution.txt[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ 保存结果失败: {str(e)}[/red]")
        
        # 绘制Pareto前沿
        console.print(f"\n[bold]生成可视化结果...[/bold]")
        try:
            plot_pareto_front([results['objectives']], [name],
                            f'{args.output_dir}/{name}_pareto_front.png')
            console.print(f"[green]✓ Pareto前沿图已保存: {args.output_dir}/{name}_pareto_front.png[/green]")
        except Exception as e:
            console.print(f"[red]✗ 绘制Pareto前沿失败: {str(e)}[/red]")
        
        # 绘制收敛曲线
        try:
            plot_convergence([results['generation_history']], [name],
                            f'{args.output_dir}/{name}_convergence.png')
            console.print(f"[green]✓ 收敛曲线已保存: {args.output_dir}/{name}_convergence.png[/green]")
        except Exception as e:
            console.print(f"[red]✗ 绘制收敛曲线失败: {str(e)}[/red]")
        
        # 绘制甘特图
        if results['pareto_front']:
            try:
                best_solution = results['pareto_front'][0]
                schedule, makespan, total_workload = decoder.decode(best_solution)
                
                console.print(f"\n[bold]绘制甘特图...[/bold]")
                console.print(f"[blue]最优解: Makespan={makespan}, Workload={total_workload}[/blue]")
                
                # 添加调试信息
                console.print(f"[yellow]调试信息: schedule类型={type(schedule)}, 长度={len(schedule) if schedule else 0}[/yellow]")
                if schedule and len(schedule) > 0:
                    console.print(f"[yellow]第一个元素: {schedule[0]}[/yellow]")
                
                plot_gantt_chart(schedule, f'{args.output_dir}/{name}_gantt_chart.png')
                console.print(f"[green]✓ 甘特图已保存: {args.output_dir}/{name}_gantt_chart.png[/green]")
            except Exception as e:
                console.print(f"[red]✗ 绘制甘特图失败: {str(e)}[/red]")
                import traceback
                console.print(f"[red]详细错误: {traceback.format_exc()}[/red]")
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]程序被用户中断[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]程序执行出错: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return 1

def save_text_results(results, filepath):
    """保存文本格式的结果"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("柔性作业车间调度多目标优化结果\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本信息
        f.write(f"算法运行时间: {results.get('runtime', 0):.2f} 秒\n")
        f.write(f"Pareto前沿大小: {len(results.get('objectives', []))}\n")
        f.write(f"总迭代次数: {len(results.get('generation_history', []))}\n\n")
        
        # Pareto前沿
        f.write("Pareto前沿解集:\n")
        f.write("-" * 30 + "\n")
        objectives = results.get('objectives', [])
        for i, obj in enumerate(objectives):
            f.write(f"解 {i+1}: Makespan = {obj[0]:.2f}, Workload = {obj[1]:.2f}\n")
        
        # 统计信息
        if objectives:
            makespans = [obj[0] for obj in objectives]
            workloads = [obj[1] for obj in objectives]
            
            f.write("\n统计信息:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Makespan - 最小值: {min(makespans):.2f}, 最大值: {max(makespans):.2f}, 平均值: {np.mean(makespans):.2f}\n")
            f.write(f"Workload - 最小值: {min(workloads):.2f}, 最大值: {max(workloads):.2f}, 平均值: {np.mean(workloads):.2f}\n")

def save_csv_results(results, filepath):
    """保存CSV格式的Pareto前沿"""
    import csv
    
    objectives = results.get('objectives', [])
    if not objectives:
        return
        
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Solution_ID', 'Makespan', 'Total_Workload'])
        
        for i, obj in enumerate(objectives):
            writer.writerow([i+1, obj[0], obj[1]])

def save_best_solution_details(results, decoder, filepath):
    """保存最优解的详细信息"""
    pareto_front = results.get('pareto_front', [])
    objectives = results.get('objectives', [])
    
    if not pareto_front or not objectives:
        return
        
    # 找到makespan最小的解作为最优解
    best_idx = np.argmin([obj[0] for obj in objectives])
    best_solution = pareto_front[best_idx]
    best_objectives = objectives[best_idx]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("最优调度方案详情\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"目标函数值:\n")
        f.write(f"  - 总流程时间 (Makespan): {best_objectives[0]:.2f}\n")
        f.write(f"  - 机器总负载 (Workload): {best_objectives[1]:.2f}\n\n")
        
        # 解码最优解
        try:
            schedule, makespan, workload = decoder.decode(best_solution)
            
            f.write("调度方案:\n")
            f.write("-" * 20 + "\n")
            
            # 按机器分组显示
            machine_operations = {}
            for op in schedule:
                machine_id = op['machine_id']
                if machine_id not in machine_operations:
                    machine_operations[machine_id] = []
                machine_operations[machine_id].append(op)
            
            # 按机器顺序输出
            for machine_id in sorted(machine_operations.keys()):
                f.write(f"\n机器 {machine_id}:\n")
                operations = sorted(machine_operations[machine_id], key=lambda x: x['start_time'])
                for op in operations:
                    f.write(f"  工序 J{op['job_id']}-O{op['operation_id']}: ")
                    f.write(f"开始时间 {op['start_time']:.1f} -> 结束时间 {op['finish_time']:.1f} ")
                    f.write(f"(加工时间 {op['processing_time']:.1f})\n")
            
            # 机器负载统计
            f.write("\n机器负载统计:\n")
            f.write("-" * 20 + "\n")
            machine_workloads = {}
            for op in schedule:
                machine_id = op['machine_id']
                machine_workloads[machine_id] = machine_workloads.get(machine_id, 0) + op['processing_time']
            
            for machine_id in sorted(machine_workloads.keys()):
                workload = machine_workloads[machine_id]
                utilization = (workload / makespan) * 100 if makespan > 0 else 0
                f.write(f"机器 {machine_id}: 负载 {workload:.2f}, 利用率 {utilization:.1f}%\n")
                
        except Exception as e:
            f.write(f"解码最优解时出错: {str(e)}\n")

if __name__ == "__main__":
    exit(main())