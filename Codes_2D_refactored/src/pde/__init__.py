from .base_pde import BasePDE
from .p1_pde import P1_PDE
from .p2_pde import P2_PDE
from .p3_pde import P3_PDE
from .p4_pde import P4_PDE

__all__ = ['BasePDE', 'P1_PDE', 'P2_PDE', 'P3_PDE', 'P4_PDE']


def get_pde(problem_name):
    """
    Factory function to get PDE class by problem name.

    Args:
        problem_name: Name of the problem ('P1', 'P2', 'P3', or 'P4')

    Returns:
        PDE class instance
    """
    pde_map = {
        'P1': P1_PDE,
        'P2': P2_PDE,
        'P3': P3_PDE,
        'P4': P4_PDE
    }

    if problem_name not in pde_map:
        raise ValueError(f"Unknown problem: {problem_name}. Choose from {list(pde_map.keys())}")

    return pde_map[problem_name]()
