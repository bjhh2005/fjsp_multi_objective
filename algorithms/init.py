"""
算法模块初始化
"""

from algorithms.nsga2 import NSGA2
from algorithms.moead import MOEAD
from algorithms.mopso import MOPSO

# 算法注册表
ALGORITHM_REGISTRY = {
    'nsga2': NSGA2,
    'moead': MOEAD,
    'mopso': MOPSO
}

def get_algorithm(algorithm_name: str):
    """
    根据算法名称获取算法类
    
    Args:
        algorithm_name: 算法名称
        
    Returns:
        算法类
    """
    if algorithm_name.lower() not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")
    
    return ALGORITHM_REGISTRY[algorithm_name.lower()]