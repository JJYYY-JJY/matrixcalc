"""
Utility functions for the Gaussian elimination solver.

This module provides helper functions for displaying matrices and
systems of equations in a user-friendly format.
"""

import numpy as np
from typing import Optional


def print_matrix(matrix: np.ndarray, title: Optional[str] = None, precision: int = 3) -> None:
    """
    Print a matrix in a nicely formatted way.
    
    Args:
        matrix: The matrix to print
        title: Optional title to display above the matrix
        precision: Number of decimal places to display
        
    Examples:
        >>> import numpy as np
        >>> A = np.array([[1, 2, 3], [4, 5, 6]])
        >>> print_matrix(A, "Matrix A")
        Matrix A:
        [  1.000   2.000   3.000 ]
        [  4.000   5.000   6.000 ]
    """
    if title:
        print(f"{title}:")
    
    if matrix.size == 0:
        print("[]")
        return
    
    # Handle both 1D and 2D arrays
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    
    # Calculate the width needed for each column
    max_width = 0
    formatted_elements = []
    
    for i in range(matrix.shape[0]):
        row_elements = []
        for j in range(matrix.shape[1]):
            element_str = f"{matrix[i, j]:.{precision}f}"
            row_elements.append(element_str)
            max_width = max(max_width, len(element_str))
        formatted_elements.append(row_elements)
    
    # Print the matrix with proper alignment
    for row in formatted_elements:
        row_str = "[ " + "  ".join(f"{elem:>{max_width}}" for elem in row) + " ]"
        print(row_str)
    
    print()  # Add blank line after matrix


def print_system(A: np.ndarray, b: np.ndarray, precision: int = 3) -> None:
    """
    Display the system Ax = b as a set of equations.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        precision: Number of decimal places to display
        
    Examples:
        >>> import numpy as np
        >>> A = np.array([[2, 3], [1, -1]], dtype=float)
        >>> b = np.array([7, 1], dtype=float)
        >>> print_system(A, b)
        System of equations:
        2.000*x₁ + 3.000*x₂ = 7.000
        1.000*x₁ - 1.000*x₂ = 1.000
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    if A.ndim != 2:
        raise ValueError("Matrix A must be 2-dimensional")
    if b.ndim != 1:
        raise ValueError("Vector b must be 1-dimensional")
    if A.shape[0] != len(b):
        raise ValueError("Incompatible dimensions")
    
    n, m = A.shape
    
    print("System of equations:")
    
    # Unicode subscripts for variable names
    subscripts = "₀₁₂₃₄₅₆₇₈₉"
    
    for i in range(n):
        equation_parts = []
        
        for j in range(m):
            coeff = A[i, j]
            
            # Format coefficient
            if j == 0:  # First term
                if coeff == 1:
                    coeff_str = ""
                elif coeff == -1:
                    coeff_str = "-"
                else:
                    coeff_str = f"{coeff:.{precision}f}*"
            else:  # Subsequent terms
                if coeff == 1:
                    coeff_str = " + "
                elif coeff == -1:
                    coeff_str = " - "
                elif coeff > 0:
                    coeff_str = f" + {coeff:.{precision}f}*"
                elif coeff < 0:
                    coeff_str = f" - {abs(coeff):.{precision}f}*"
                else:  # coeff == 0
                    continue
            
            # Create variable name with subscript
            if j < 10:
                var_name = f"x{subscripts[j]}"
            else:
                var_name = f"x_{j}"
            
            if j == 0 and coeff_str in ["", "-"]:
                equation_parts.append(f"{coeff_str}{var_name}")
            elif coeff_str.endswith("*"):
                equation_parts.append(f"{coeff_str}{var_name}")
            else:
                equation_parts.append(f"{coeff_str}{var_name}")
        
        # Handle the case where all coefficients are zero
        if not equation_parts:
            equation_str = "0"
        else:
            equation_str = "".join(equation_parts)
        
        # Add right-hand side
        rhs = f"{b[i]:.{precision}f}"
        print(f"{equation_str} = {rhs}")
    
    print()  # Add blank line after system


def print_augmented_matrix(A: np.ndarray, b: np.ndarray, title: Optional[str] = None, precision: int = 3) -> None:
    """
    Print the augmented matrix [A|b] with a visual separator.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        title: Optional title to display above the matrix
        precision: Number of decimal places to display
        
    Examples:
        >>> import numpy as np
        >>> A = np.array([[2, 3], [1, -1]], dtype=float)
        >>> b = np.array([7, 1], dtype=float)
        >>> print_augmented_matrix(A, b, "Augmented Matrix")
        Augmented Matrix:
        [  2.000   3.000 |   7.000 ]
        [  1.000  -1.000 |   1.000 ]
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    if A.ndim != 2:
        raise ValueError("Matrix A must be 2-dimensional")
    if b.ndim != 1:
        raise ValueError("Vector b must be 1-dimensional")
    if A.shape[0] != len(b):
        raise ValueError("Incompatible dimensions")
    
    if title:
        print(f"{title}:")
    
    n, m = A.shape
    
    # Calculate the width needed for each column
    max_width_A = 0
    max_width_b = 0
    
    # Format all elements to determine maximum width
    formatted_A = []
    formatted_b = []
    
    for i in range(n):
        row_A = []
        for j in range(m):
            element_str = f"{A[i, j]:.{precision}f}"
            row_A.append(element_str)
            max_width_A = max(max_width_A, len(element_str))
        formatted_A.append(row_A)
        
        b_str = f"{b[i]:.{precision}f}"
        formatted_b.append(b_str)
        max_width_b = max(max_width_b, len(b_str))
    
    # Print the augmented matrix with separator
    for i in range(n):
        # Print A part
        A_part = "  ".join(f"{elem:>{max_width_A}}" for elem in formatted_A[i])
        # Print separator and b part
        b_part = f"{formatted_b[i]:>{max_width_b}}"
        print(f"[ {A_part} | {b_part} ]")
    
    print()  # Add blank line after matrix


def format_solution(solution: np.ndarray, precision: int = 6) -> str:
    """
    Format the solution vector for display.
    
    Args:
        solution: The solution vector
        precision: Number of decimal places to display
        
    Returns:
        str: Formatted solution string
        
    Examples:
        >>> import numpy as np
        >>> x = np.array([1.0, 2.5, -0.333333])
        >>> print(format_solution(x))
        x₁ = 1.000000, x₂ = 2.500000, x₃ = -0.333333
    """
    if solution.size == 0:
        return "No solution or empty solution set"
    
    # Unicode subscripts for variable names
    subscripts = "₀₁₂₃₄₅₆₇₈₉"
    
    solution_parts = []
    for i, value in enumerate(solution):
        if i < 10:
            var_name = f"x{subscripts[i]}"
        else:
            var_name = f"x_{i}"
        
        solution_parts.append(f"{var_name} = {value:.{precision}f}")
    
    return ", ".join(solution_parts) 