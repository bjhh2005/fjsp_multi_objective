"""
命令行参数解析工具
"""

import argparse

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
    
    # 最优解选择参数
    parser.add_argument('--best_criteria', type=str, default='makespan',
                        choices=['makespan', 'workload', 'balanced'],
                        help='最优解选择标准 (makespan, workload, balanced)')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果输出目录')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出')
    
    return parser.parse_args()

def get_algorithm_params(algorithm_name, args):
    """获取算法参数"""
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
    
    return params