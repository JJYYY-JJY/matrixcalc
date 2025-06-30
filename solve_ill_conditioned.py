#!/usr/bin/env python3
"""
Practical Solution for Your 9×9 Ill-Conditioned System

This script provides the most effective methods for solving severely ill-conditioned
linear systems, specifically designed for your use case.
"""

import numpy as np
import warnings

# Use numpy's solver if scipy is not available
try:
    from scipy.linalg import solve as scipy_solve
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_solve = np.linalg.solve
    SCIPY_AVAILABLE = False
    print("Note: scipy not available, using numpy.linalg.solve (still uses LAPACK)")

def solve_ill_conditioned_system(A, b, method='auto', verbose=True):
    """
    Solve an ill-conditioned linear system using the most appropriate method.
    
    Args:
        A: 9×9 coefficient matrix
        b: Right-hand side vector
        method: 'auto', 'equilibrated_float128', 'float128', 'equilibrated', 'standard'
        verbose: Print detailed information
        
    Returns:
        dict: Solution and diagnostics
    """
    
    if verbose:
        condition_number = np.linalg.cond(A)
        print(f"System Analysis:")
        print(f"  Condition number: {condition_number:.2e}")
        print(f"  Matrix elements range: [{np.min(A):.2e}, {np.max(A):.2e}]")
        print(f"  Severity: {'EXTREMELY' if condition_number > 1e10 else 'MODERATELY'} ill-conditioned")
        print()
    
    if method == 'auto':
        # Automatically choose best method based on condition number
        condition_number = np.linalg.cond(A)
        if condition_number > 1e10:
            method = 'equilibrated_float128'
        elif condition_number > 1e6:
            method = 'float128'
        else:
            method = 'equilibrated'
        
        if verbose:
            print(f"Auto-selected method: {method}")
    
    # Method implementations
    if method == 'equilibrated_float128':
        return solve_equilibrated_float128(A, b, verbose)
    elif method == 'float128':
        return solve_float128(A, b, verbose)
    elif method == 'equilibrated':
        return solve_equilibrated(A, b, verbose)
    else:
        return solve_standard(A, b, verbose)

def equilibrate_matrix(A, b):
    """Equilibrate matrix to improve condition number."""
    # Compute scaling factors
    row_scales = 1.0 / np.sqrt(np.sum(A**2, axis=1))
    col_scales = 1.0 / np.sqrt(np.sum(A**2, axis=0))
    
    # Avoid division by zero
    row_scales = np.where(np.isfinite(row_scales), row_scales, 1.0)
    col_scales = np.where(np.isfinite(col_scales), col_scales, 1.0)
    
    # Apply scaling
    A_eq = np.diag(row_scales) @ A @ np.diag(col_scales)
    b_eq = row_scales * b
    
    return A_eq, b_eq, row_scales, col_scales

def solve_equilibrated_float128(A, b, verbose=True):
    """RECOMMENDED: Equilibration + Higher Precision - Best for extremely ill-conditioned systems."""
    if verbose:
        print("Using: Matrix Equilibration + numpy.float128 (RECOMMENDED)")
        print("Benefits: Improved conditioning + higher precision arithmetic")
    
    # Step 1: Equilibrate
    A_eq, b_eq, row_scales, col_scales = equilibrate_matrix(A, b)
    
    if verbose:
        original_cond = np.linalg.cond(A)
        equilibrated_cond = np.linalg.cond(A_eq)
        print(f"  Condition number improvement: {original_cond/equilibrated_cond:.1f}x")
    
    # Step 2: Solve with higher precision
    A_hp = A_eq.astype(np.float128)
    b_hp = b_eq.astype(np.float128)
    
    y_hp = np.linalg.solve(A_hp, b_hp)
    
    # Step 3: Recover solution
    x = col_scales * y_hp.astype(np.float64)
    
    return evaluate_and_return_solution(A, b, x, "Equilibrated + float128", verbose)

def solve_float128(A, b, verbose=True):
    """Higher precision arithmetic without equilibration."""
    if verbose:
        print("Using: numpy.float128 (Higher Precision)")
        print("Benefits: ~4x more precision than standard double precision")
    
    A_hp = A.astype(np.float128)
    b_hp = b.astype(np.float128)
    
    x_hp = np.linalg.solve(A_hp, b_hp)
    x = x_hp.astype(np.float64)
    
    return evaluate_and_return_solution(A, b, x, "float128", verbose)

def solve_equilibrated(A, b, verbose=True):
    """Matrix equilibration with standard precision."""
    if verbose:
        print("Using: Matrix Equilibration + Standard Precision")
        print("Benefits: Improved conditioning with minimal computational overhead")
    
    A_eq, b_eq, row_scales, col_scales = equilibrate_matrix(A, b)
    
    if verbose:
        original_cond = np.linalg.cond(A)
        equilibrated_cond = np.linalg.cond(A_eq)
        print(f"  Condition number improvement: {original_cond/equilibrated_cond:.1f}x")
    
    y_eq = scipy_solve(A_eq, b_eq)
    x = col_scales * y_eq
    
    return evaluate_and_return_solution(A, b, x, "Equilibrated", verbose)

def solve_standard(A, b, verbose=True):
    """Standard scipy solver (LAPACK with partial pivoting)."""
    if verbose:
        print("Using: Standard SciPy solver (LAPACK)")
        print("Benefits: Fast, reliable for well-conditioned systems")
    
    x = scipy_solve(A, b)
    return evaluate_and_return_solution(A, b, x, "Standard", verbose)

def evaluate_and_return_solution(A, b, x, method_name, verbose=True):
    """Evaluate solution quality and return results."""
    # Compute diagnostics
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual)
    max_residual = np.max(np.abs(residual))
    
    # Determine solution quality
    if residual_norm < 1e-10:
        quality = "EXCELLENT"
    elif residual_norm < 1e-6:
        quality = "GOOD"
    elif residual_norm < 1e-3:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"
    
    if verbose:
        print(f"\nResults ({method_name}):")
        print(f"  Residual norm ||Ax - b||: {residual_norm:.2e}")
        print(f"  Max residual component: {max_residual:.2e}")
        print(f"  Solution quality: {quality}")
        
        if residual_norm > 1e-6:
            print(f"  ⚠️  Warning: Large numerical errors detected!")
        else:
            print(f"  ✅ Solution appears accurate")
    
    return {
        'solution': x,
        'residual_norm': residual_norm,
        'max_residual': max_residual,
        'quality': quality,
        'method': method_name
    }

def demonstrate_with_sample_system():
    """Demonstrate with a sample severely ill-conditioned 9×9 system."""
    print("🧮 DEMONSTRATION: Solving Severely Ill-Conditioned 9×9 System")
    print("=" * 70)
    
    # Create a sample system similar to your described problem
    np.random.seed(42)
    A = np.random.randn(9, 9)
    
    # Make it severely ill-conditioned with varying scales (1 to 10^12)
    scales = np.logspace(0, 12, 9)
    A = A * scales[np.newaxis, :]
    
    # Add some structure
    for i in range(9):
        for j in range(9):
            if abs(i - j) <= 1:
                A[i, j] *= 10
    
    # Create RHS
    x_true = np.ones(9)  # Known solution for verification
    b = A @ x_true
    
    print("Sample system created with:")
    print(f"  Matrix elements ranging from {np.min(A):.2e} to {np.max(A):.2e}")
    print(f"  Condition number: {np.linalg.cond(A):.2e}")
    print()
    
    # Test different methods
    methods = ['standard', 'equilibrated', 'float128', 'equilibrated_float128']
    results = {}
    
    for method in methods:
        print(f"\n{'-'*50}")
        result = solve_ill_conditioned_system(A, b, method=method, verbose=True)
        results[method] = result
        
        # Compute error against known solution
        error = np.linalg.norm(result['solution'] - x_true)
        print(f"  Error vs true solution: {error:.2e}")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Residual':<12} {'Error':<12} {'Quality':<12}")
    print("-" * 60)
    
    for method, result in results.items():
        error = np.linalg.norm(result['solution'] - x_true)
        print(f"{method:<25} {result['residual_norm']:<12.2e} {error:<12.2e} {result['quality']:<12}")

# Template for your actual system
def solve_your_system():
    """
    TEMPLATE: Replace with your actual 9×9 system.
    
    Instructions:
    1. Replace the A matrix below with your actual coefficient matrix
    2. Replace the b vector below with your actual right-hand side
    3. Run this function to get the best solution for your system
    """
    
    # TODO: Replace these with your actual system
    # Example format:
    A = np.array([
        [2.1e0,  1.5e3,  2.3e6,  4.7e9,  1.1e12, 3.3e2,  7.8e5,  2.1e8,  9.4e11],
        [1.3e1,  4.2e4,  8.9e7,  2.1e10, 5.6e12, 1.7e3,  9.8e6,  4.3e9,  1.2e12],
        # ... (continue with your actual matrix values)
        # ... (7 more rows)
    ])
    
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])  # Your actual RHS
    
    # If you don't have your actual system yet, use the demonstration
    if A.shape != (9, 9):
        print("⚠️  Please replace A and b with your actual 9×9 system")
        print("Running demonstration instead...")
        demonstrate_with_sample_system()
        return
    
    print("🎯 SOLVING YOUR ACTUAL SYSTEM")
    print("=" * 50)
    
    # Use the automatic method selection (recommended)
    result = solve_ill_conditioned_system(A, b, method='auto', verbose=True)
    
    print(f"\n🎉 FINAL SOLUTION:")
    print(f"x = {result['solution']}")
    print(f"\nSolution quality: {result['quality']}")
    print(f"Method used: {result['method']}")
    
    # Verification
    verification = A @ result['solution']
    print(f"\nVerification (A @ x):")
    print(f"Computed: {verification}")
    print(f"Expected: {b}")
    print(f"Difference: {verification - b}")
    
    return result

if __name__ == "__main__":
    # Run demonstration
    demonstrate_with_sample_system()
    
    print(f"\n{'='*70}")
    print("TO USE WITH YOUR ACTUAL SYSTEM:")
    print("1. Edit the solve_your_system() function above")
    print("2. Replace A and b with your actual matrix and vector")
    print("3. Call solve_your_system()")
    print(f"{'='*70}")
    
    # Uncomment this line after setting up your actual system:
    # solve_your_system() 