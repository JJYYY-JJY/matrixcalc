"""
Core implementation of Gaussian elimination with partial pivoting.

This module contains the main solve function that implements the Gaussian
elimination algorithm for solving linear systems Ax = b.
"""

import numpy as np
from typing import Tuple


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the linear system Ax = b using Gaussian elimination with partial pivoting.
    
    This function now supports rectangular matrices:
    - n < m (underdetermined): May have infinite solutions
    - n > m (overdetermined): May have no solution or unique solution  
    - n = m (square): May have unique solution, no solution, or infinite solutions
    
    Args:
        A: Coefficient matrix (n x m numpy array, any size)
        b: Right-hand side vector (n x 1 numpy array)
        
    Returns:
        np.ndarray: Solution vector x (length m)
        
    Raises:
        ValueError: If the system has no solution or infinite solutions
        ValueError: If input dimensions are incompatible
        
    Examples:
        >>> import numpy as np
        >>> # Square system
        >>> A = np.array([[2, 1], [1, 1]], dtype=float)
        >>> b = np.array([3, 2], dtype=float)
        >>> x = solve(A, b)
        >>> print(x)
        [1. 1.]
        
        >>> # Overdetermined system (more equations than unknowns)
        >>> A = np.array([[1, 1], [1, 2], [2, 3]], dtype=float)
        >>> b = np.array([3, 4, 7], dtype=float)
        >>> x = solve(A, b)  # Will find least squares solution if consistent
    """
    # Input validation
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    if A.ndim != 2:
        raise ValueError("Matrix A must be 2-dimensional")
    
    if b.ndim != 1:
        raise ValueError("Vector b must be 1-dimensional")
    
    n, m = A.shape
    if len(b) != n:
        raise ValueError(f"Incompatible dimensions: A is {n}x{m}, b has length {len(b)}")
    
    if n == 0 or m == 0:
        if n == 0 and m == 0:
            return np.array([])
        else:
            raise ValueError("Empty matrix or vector not supported for non-trivial cases")
    
    # Create augmented matrix [A|b]
    augmented = np.column_stack([A, b])
    
    # Forward elimination with partial pivoting
    _forward_elimination(augmented)
    
    # Check for no solution or infinite solutions
    _check_solution_existence(augmented)
    
    # Back substitution to find the solution
    return _back_substitution(augmented)


def _forward_elimination(augmented: np.ndarray) -> None:
    """
    Perform forward elimination with partial pivoting on the augmented matrix.
    
    Now supports rectangular matrices by performing elimination up to min(n, m) steps.
    
    Args:
        augmented: The augmented matrix [A|b] to be transformed
    """
    n, total_cols = augmented.shape
    m = total_cols - 1  # Number of variables (exclude b column)
    
    # Perform elimination for min(n, m) steps
    num_pivots = min(n, m)
    
    for i in range(num_pivots):
        # Partial pivoting: find the row with the largest absolute value in column i
        # starting from row i onwards
        pivot_row = i + np.argmax(np.abs(augmented[i:, i]))
        
        # Swap rows if necessary
        if pivot_row != i:
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
        
        # Check if pivot is essentially zero
        if np.abs(augmented[i, i]) < 1e-14:
            continue  # Skip this column, will be handled in solution checking
        
        # Eliminate column i in all rows below row i
        for j in range(i + 1, n):
            if np.abs(augmented[j, i]) > 1e-14:  # Avoid operations on essentially zero elements
                factor = augmented[j, i] / augmented[i, i]
                augmented[j, i:] -= factor * augmented[i, i:]


def _check_solution_existence(augmented: np.ndarray) -> None:
    """
    Check if the system has a unique solution, no solution, or infinite solutions.
    
    Now handles rectangular matrices by checking rank conditions properly.
    
    Args:
        augmented: The augmented matrix after forward elimination
        
    Raises:
        ValueError: If the system has no solution or infinite solutions
    """
    n, total_cols = augmented.shape
    m = total_cols - 1  # Number of variables (exclude b column)
    
    # First, check for inconsistency (no solution)
    # This occurs when we have a row like [0, 0, ..., 0, c] where c != 0
    for i in range(n):
        if np.allclose(augmented[i, :-1], 0, atol=1e-14) and not np.allclose(augmented[i, -1], 0, atol=1e-14):
            raise ValueError("System has no solution (inconsistent system)")
    
    # Count the number of non-zero rows (this gives us the rank)
    rank_A = 0
    rank_augmented = 0
    
    for i in range(n):
        if not np.allclose(augmented[i, :-1], 0, atol=1e-14):
            rank_A += 1
        if not np.allclose(augmented[i, :], 0, atol=1e-14):
            rank_augmented += 1
    
    # Check for inconsistency using rank condition
    if rank_augmented > rank_A:
        raise ValueError("System has no solution (inconsistent system)")
    
    # Check for underdetermined system (infinite solutions)
    # This happens when rank(A) < number of variables
    if rank_A < m:
        raise ValueError("System has infinite solutions (underdetermined system)")
    
    # For overdetermined systems (n > m), if we get here, the system is consistent
    # and has a unique solution in the least squares sense
    
    # For square systems (n == m), check if we have full rank
    if n == m and rank_A < n:
        raise ValueError("System has infinite solutions (rank deficient matrix)")


def _back_substitution(augmented: np.ndarray) -> np.ndarray:
    """
    Perform back substitution to solve for the variables.
    
    Now handles rectangular matrices by solving for m variables using 
    the available pivot equations.
    
    Args:
        augmented: The augmented matrix in row echelon form
        
    Returns:
        np.ndarray: The solution vector x (length m)
    """
    n, total_cols = augmented.shape
    m = total_cols - 1  # Number of variables (exclude b column)
    
    x = np.zeros(m)
    
    # Determine how many pivots we actually have
    num_pivots = min(n, m)
    
    # Start from the last pivot and work backwards
    for i in range(num_pivots - 1, -1, -1):
        # Skip if this row is essentially zero (no pivot)
        if np.abs(augmented[i, i]) < 1e-14:
            continue
            
        # Calculate the sum of known variables
        sum_known = np.dot(augmented[i, i + 1:m], x[i + 1:m])
        
        # Solve for x[i]
        x[i] = (augmented[i, -1] - sum_known) / augmented[i, i]
    
    return x


def get_solution_info(A: np.ndarray, b: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Get information about the solution without raising exceptions.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        
    Returns:
        Tuple[str, np.ndarray]: (solution_type, solution_or_empty_array)
        where solution_type is one of: "unique", "no_solution", "infinite_solutions"
    """
    try:
        solution = solve(A, b)
        return "unique", solution
    except ValueError as e:
        error_msg = str(e).lower()
        if "no solution" in error_msg:
            return "no_solution", np.array([])
        elif "infinite solutions" in error_msg:
            return "infinite_solutions", np.array([])
        else:
            return "error", np.array([]) 