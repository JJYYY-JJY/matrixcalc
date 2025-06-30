"""
Gaussian Elimination Solver Package

This package provides a robust implementation of Gaussian elimination
with partial pivoting for solving systems of linear equations Ax = b.

Main function:
    solve(A, b): Solve the linear system Ax = b

Example:
    >>> import numpy as np
    >>> from gaussian_elimination import solve
    >>> A = np.array([[2, 1], [1, 1]], dtype=float)
    >>> b = np.array([3, 2], dtype=float)
    >>> solution = solve(A, b)
    >>> print(solution)
    [1. 1.]
"""

from .core import solve
from .utils import print_matrix, print_system, format_solution

__version__ = "1.0.0"
__author__ = "Gaussian Elimination Solver Team"

__all__ = ["solve", "print_matrix", "print_system", "format_solution"] 