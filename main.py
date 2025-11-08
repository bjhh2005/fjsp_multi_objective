"""
主程序入口
支持通过命令行参数选择算法
"""

import time
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from data import FJSPProblem
from utils.decoder import FJSPDecoder
from algorithms.init import get_algorithm
from utils.visualization import plot_pareto_front, plot_gantt_chart, plot_convergence

# 导入工具模块
from utils.args_parser import parse_arguments, get_algorithm_params
from utils.results_saver import save_all_results
from utils.solution_selector import get_best_solution, get_selection_criteria_description

def run_single_algorithm(algorithm_name, problem, args, console):
    """运行单个算法"""
    
    console.print(f"\n[bold cyan]开始运行 {algorithm_name.upper()} 算法...[/bold cyan]")
    
    # 准备算法参数
    params = get_algorithm_params(algorithm_name, args)
    
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
        
        # 显示算法配置
        config_info = f"""
        [cyan]算法配置:[/cyan]
        • 算法: {args.algorithm.upper()}
        • 种群大小: {args.pop_size}
        • 最大迭代: {args.max_gen}
        • 最优解标准: {get_selection_criteria_description(args.best_criteria)}
        """
        console.print(Panel(config_info.strip(), title="运行配置", expand=False))
        
        # 运行单个算法
        name, results, error = run_single_algorithm(args.algorithm, problem, args, console)
        
        if error:
            console.print(f"\n[bold red]算法执行失败: {error}[/bold red]")
            return 1
        
        if not results or not results.get('pareto_front'):
            console.print(f"\n[bold red]未获得有效结果[/bold red]")
            return 1
        
        # 获取最优解
        best_solution, best_schedule, best_makespan, best_workload, criteria_desc = get_best_solution(
            results, decoder, selection_criteria=args.best_criteria
        )
        
        console.print(f"\n[bold]最优解信息 ({criteria_desc}):[/bold]")
        console.print(f"[blue]• Makespan: {best_makespan}[/blue]")
        console.print(f"[blue]• Workload: {best_workload}[/blue]")
        
        # 保存所有结果
        console.print(f"\n[bold]保存结果...[/bold]")
        success, message = save_all_results(
            results, decoder, best_solution, best_schedule,
            best_makespan, best_workload, args.best_criteria,
            name, args.output_dir
        )
        
        if success:
            console.print(f"[green]✓ {message}[/green]")
        else:
            console.print(f"[red]✗ {message}[/red]")
        
        # 生成可视化结果
        console.print(f"\n[bold]生成可视化结果...[/bold]")
        try:
            plot_pareto_front([results['objectives']], [name],
                            f'{args.output_dir}/{name}_pareto_front.png')
            console.print(f"[green]✓ Pareto前沿图已保存[/green]")
        except Exception as e:
            console.print(f"[red]✗ 绘制Pareto前沿失败: {str(e)}[/red]")
        
        try:
            plot_convergence([results['generation_history']], [name],
                            f'{args.output_dir}/{name}_convergence.png')
            console.print(f"[green]✓ 收敛曲线已保存[/green]")
        except Exception as e:
            console.print(f"[red]✗ 绘制收敛曲线失败: {str(e)}[/red]")
        
        # 绘制甘特图
        if best_schedule:
            try:
                plot_gantt_chart(best_schedule, f'{args.output_dir}/{name}_gantt_chart.png')
                console.print(f"[green]✓ 甘特图已保存[/green]")
            except Exception as e:
                console.print(f"[red]✗ 绘制甘特图失败: {str(e)}[/red]")
        else:
            console.print(f"[yellow]⚠ 无法绘制甘特图: 最优解解码失败[/yellow]")
        
        console.print(f"\n[bold green]🎉 所有任务完成! 结果保存在 '{args.output_dir}' 目录[/bold green]")
        return 0
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]程序被用户中断[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]程序执行出错: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return 1

if __name__ == "__main__":
    exit(main())