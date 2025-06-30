#!/usr/bin/env python3
"""
Comprehensive Analysis of Numerical Strategies for Ill-Conditioned Linear Systems

This script demonstrates various approaches to solve severely ill-conditioned 9x9 systems
with elements ranging from single digits to 10^12, comparing their accuracy and reliability.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.linalg import solve as scipy_solve
import warnings
from typing import Tuple, Dict, Any
import time

# Suppress overflow warnings for demonstration
warnings.filterwarnings('ignore', category=RuntimeWarning)

def create_ill_conditioned_system(n: int = 9, condition_target: float = 1e12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a severely ill-conditioned n×n linear system.
    
    Args:
        n: Size of the system
        condition_target: Target condition number
        
    Returns:
        Tuple of (A, b, x_true) where Ax = b and x_true is the known solution
    """
    np.random.seed(42)  # For reproducibility
    
    # Create a matrix with widely varying scales
    A = np.random.randn(n, n)
    
    # Make it severely ill-conditioned by scaling columns differently
    scales = np.logspace(0, 12, n)  # From 1 to 10^12
    A = A * scales[np.newaxis, :]
    
    # Add some structure to make it more realistic
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= 1:  # Emphasize diagonal and near-diagonal
                A[i, j] *= 10
    
    # Create a known solution
    x_true = np.ones(n)
    b = A @ x_true
    
    return A, b, x_true

def print_system_info(A: np.ndarray, b: np.ndarray, title: str = "System Information"):
    """Print detailed information about the linear system."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Matrix size: {A.shape[0]}×{A.shape[1]}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    print(f"Matrix elements range: [{np.min(A):.2e}, {np.max(A):.2e}]")
    print(f"Matrix norm: {np.linalg.norm(A):.2e}")
    print(f"RHS norm: {np.linalg.norm(b):.2e}")

def evaluate_solution(A: np.ndarray, b: np.ndarray, x: np.ndarray, x_true: np.ndarray, method_name: str):
    """Evaluate and print solution quality metrics."""
    residual = A @ x - b
    error = x - x_true
    
    residual_norm = np.linalg.norm(residual)
    error_norm = np.linalg.norm(error)
    relative_error = error_norm / np.linalg.norm(x_true)
    
    print(f"\n{method_name} Results:")
    print(f"  Residual norm ||Ax - b||: {residual_norm:.2e}")
    print(f"  Error norm ||x - x_true||: {error_norm:.2e}")
    print(f"  Relative error: {relative_error:.2e}")
    print(f"  Max residual component: {np.max(np.abs(residual)):.2e}")
    
    return {
        'residual_norm': residual_norm,
        'error_norm': error_norm,
        'relative_error': relative_error,
        'solution': x.copy()
    }

# ============================================================================
# STRATEGY 1: ROBUST DIRECT METHOD WITH PARTIAL PIVOTING
# ============================================================================

def gaussian_elimination_partial_pivoting(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Implement Gaussian elimination with partial pivoting for better numerical stability.
    
    Partial pivoting selects the largest absolute value element in each column as the pivot,
    which helps prevent division by small numbers and reduces numerical errors.
    """
    A_work = A.astype(np.float64).copy()
    b_work = b.astype(np.float64).copy()
    n = len(b_work)
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find the row with the largest absolute value in column i (from row i onwards)
        pivot_row = i + np.argmax(np.abs(A_work[i:, i]))
        
        # Swap rows if necessary
        if pivot_row != i:
            A_work[[i, pivot_row]] = A_work[[pivot_row, i]]
            b_work[[i, pivot_row]] = b_work[[pivot_row, i]]
        
        # Check for near-zero pivot
        if abs(A_work[i, i]) < 1e-14:
            print(f"Warning: Very small pivot {A_work[i, i]:.2e} at position {i}")
        
        # Eliminate column i in rows below
        for j in range(i + 1, n):
            if abs(A_work[j, i]) > 1e-14:
                factor = A_work[j, i] / A_work[i, i]
                A_work[j, i:] -= factor * A_work[i, i:]
                b_work[j] -= factor * b_work[i]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b_work[i] - np.dot(A_work[i, i + 1:], x[i + 1:])) / A_work[i, i]
    
    return x

def test_partial_pivoting(A: np.ndarray, b: np.ndarray, x_true: np.ndarray) -> Dict[str, Any]:
    """Test Gaussian elimination with partial pivoting."""
    print(f"\n{'='*60}")
    print("STRATEGY 1: GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING")
    print(f"{'='*60}")
    print("""
    WHY PARTIAL PIVOTING IS ESSENTIAL:
    
    1. NUMERICAL STABILITY: In ill-conditioned matrices, small pivots can lead to 
       catastrophic error amplification. Partial pivoting selects the largest 
       available pivot, reducing this risk.
    
    2. ERROR CONTROL: By avoiding division by small numbers, we prevent the 
       explosive growth of rounding errors that occurs in standard elimination.
    
    3. IMPROVED ACCURACY: The row swapping ensures that the elimination process 
       maintains better numerical properties throughout the computation.
    
    4. ROBUSTNESS: Handles near-singular matrices more gracefully than naive 
       Gaussian elimination.
    """)
    
    start_time = time.time()
    x_pivot = gaussian_elimination_partial_pivoting(A, b)
    pivot_time = time.time() - start_time
    
    result = evaluate_solution(A, b, x_pivot, x_true, "Partial Pivoting")
    result['computation_time'] = pivot_time
    
    # Compare with scipy's implementation (which uses LAPACK with pivoting)
    start_time = time.time()
    x_scipy = scipy_solve(A, b)
    scipy_time = time.time() - start_time
    
    scipy_result = evaluate_solution(A, b, x_scipy, x_true, "SciPy (LAPACK)")
    print(f"  Computation time: {scipy_time:.4f} seconds")
    
    print(f"\nPartial Pivoting computation time: {pivot_time:.4f} seconds")
    
    return result

# ============================================================================
# STRATEGY 2: MATRIX PREPROCESSING/CONDITIONING
# ============================================================================

def equilibrate_matrix(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Equilibrate matrix to improve condition number using row and column scaling.
    
    This implements diagonal scaling to balance the matrix elements and reduce
    the condition number, which can significantly improve numerical stability.
    """
    A_eq = A.copy()
    b_eq = b.copy()
    n = A.shape[0]
    
    # Compute row and column scaling factors
    row_scales = 1.0 / np.sqrt(np.sum(A_eq**2, axis=1))
    col_scales = 1.0 / np.sqrt(np.sum(A_eq**2, axis=0))
    
    # Avoid division by zero
    row_scales = np.where(np.isfinite(row_scales), row_scales, 1.0)
    col_scales = np.where(np.isfinite(col_scales), col_scales, 1.0)
    
    # Apply scaling
    A_eq = np.diag(row_scales) @ A_eq @ np.diag(col_scales)
    b_eq = row_scales * b_eq
    
    return A_eq, b_eq, row_scales, col_scales

def precondition_jacobi(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Jacobi preconditioning (diagonal scaling).
    """
    D_inv = 1.0 / np.diag(A)
    D_inv = np.where(np.isfinite(D_inv), D_inv, 1.0)
    
    A_precond = np.diag(D_inv) @ A
    b_precond = D_inv * b
    
    return A_precond, b_precond, D_inv

def test_matrix_conditioning(A: np.ndarray, b: np.ndarray, x_true: np.ndarray) -> Dict[str, Any]:
    """Test matrix conditioning approaches."""
    print(f"\n{'='*60}")
    print("STRATEGY 2: MATRIX PREPROCESSING/CONDITIONING")
    print(f"{'='*60}")
    print("""
    MATRIX EQUILIBRATION THEORY:
    
    1. PURPOSE: Reduce the condition number by balancing row and column scales
    2. METHOD: Apply diagonal scaling matrices D₁ and D₂ such that D₁AD₂ has 
       better numerical properties
    3. GOAL: Make matrix elements more uniform in magnitude
    4. RECOVERY: After solving D₁AD₂y = D₁b, recover x = D₂y
    """)
    
    # Original condition number
    original_cond = np.linalg.cond(A)
    print(f"Original condition number: {original_cond:.2e}")
    
    # Test equilibration
    A_eq, b_eq, row_scales, col_scales = equilibrate_matrix(A, b)
    equilibrated_cond = np.linalg.cond(A_eq)
    
    print(f"Equilibrated condition number: {equilibrated_cond:.2e}")
    print(f"Condition number improvement: {original_cond/equilibrated_cond:.2f}x")
    
    # Solve equilibrated system
    y_eq = scipy_solve(A_eq, b_eq)
    x_eq = col_scales * y_eq  # Recover original solution
    
    result_eq = evaluate_solution(A, b, x_eq, x_true, "Equilibrated System")
    
    # Test Jacobi preconditioning
    A_jac, b_jac, D_inv = precondition_jacobi(A, b)
    jacobi_cond = np.linalg.cond(A_jac)
    
    print(f"Jacobi preconditioned condition number: {jacobi_cond:.2e}")
    print(f"Jacobi improvement: {original_cond/jacobi_cond:.2f}x")
    
    x_jac = scipy_solve(A_jac, b_jac)
    result_jac = evaluate_solution(A, b, x_jac, x_true, "Jacobi Preconditioned")
    
    return {
        'equilibrated': result_eq,
        'jacobi': result_jac,
        'condition_improvements': {
            'original': original_cond,
            'equilibrated': equilibrated_cond,
            'jacobi': jacobi_cond
        }
    }

# ============================================================================
# STRATEGY 3: HIGHER-PRECISION ARITHMETIC
# ============================================================================

def solve_higher_precision_numpy(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve using numpy's extended precision (float128)."""
    A_hp = A.astype(np.float128)
    b_hp = b.astype(np.float128)
    
    # Use numpy's linear algebra solver with higher precision
    x_hp = np.linalg.solve(A_hp, b_hp)
    
    return x_hp.astype(np.float64)  # Convert back for comparison

def solve_arbitrary_precision_mpmath(A: np.ndarray, b: np.ndarray, precision_bits: int = 100) -> np.ndarray:
    """
    Solve using mpmath for arbitrary precision arithmetic.
    
    Args:
        A, b: Input system
        precision_bits: Number of bits for precision (default 100 ≈ 30 decimal digits)
    """
    try:
        import mpmath as mp
    except ImportError:
        print("mpmath not available. Install with: pip install mpmath")
        return np.zeros_like(b)
    
    # Set precision
    mp.mp.dps = precision_bits // 3.32  # Convert bits to decimal places approximately
    
    # Convert to mpmath matrices
    A_mp = mp.matrix(A.tolist())
    b_mp = mp.matrix(b.tolist())
    
    # Solve using mpmath
    x_mp = mp.lu_solve(A_mp, b_mp)
    
    # Convert back to numpy
    x_result = np.array([float(x_mp[i]) for i in range(len(b))])
    
    return x_result

def test_higher_precision(A: np.ndarray, b: np.ndarray, x_true: np.ndarray) -> Dict[str, Any]:
    """Test higher precision arithmetic approaches."""
    print(f"\n{'='*60}")
    print("STRATEGY 3: HIGHER-PRECISION ARITHMETIC")
    print(f"{'='*60}")
    print("""
    PRECISION ARITHMETIC BENEFITS:
    
    1. EXTENDED PRECISION (float128): Uses ~19 significant decimal digits vs 
       ~15 for float64, reducing rounding errors in intermediate calculations.
    
    2. ARBITRARY PRECISION (mpmath): Allows setting precision to any desired 
       level, effectively eliminating rounding errors for well-conditioned 
       sub-problems.
    
    3. TRADE-OFFS: Higher precision comes with computational cost but can be 
       essential for extremely ill-conditioned systems.
    
    4. LIMITATIONS: Cannot fix fundamental ill-conditioning, but reduces the 
       impact of finite precision arithmetic.
    """)
    
    # Test float128
    print("Testing numpy.float128 (extended precision)...")
    start_time = time.time()
    x_float128 = solve_higher_precision_numpy(A, b)
    float128_time = time.time() - start_time
    
    result_float128 = evaluate_solution(A, b, x_float128, x_true, "NumPy float128")
    result_float128['computation_time'] = float128_time
    
    # Test mpmath if available
    print("\nTesting mpmath (arbitrary precision)...")
    start_time = time.time()
    x_mpmath = solve_arbitrary_precision_mpmath(A, b, precision_bits=100)
    mpmath_time = time.time() - start_time
    
    if not np.allclose(x_mpmath, 0):  # Check if mpmath was available
        result_mpmath = evaluate_solution(A, b, x_mpmath, x_true, "mpmath (100-bit)")
        result_mpmath['computation_time'] = mpmath_time
    else:
        result_mpmath = None
        print("mpmath not available - skipping arbitrary precision test")
    
    return {
        'float128': result_float128,
        'mpmath': result_mpmath
    }

# ============================================================================
# STRATEGY 4: ITERATIVE METHODS
# ============================================================================

def check_diagonal_dominance(A: np.ndarray) -> Tuple[bool, float]:
    """
    Check if matrix is diagonally dominant and compute dominance ratio.
    
    A matrix is diagonally dominant if |A[i,i]| >= sum(|A[i,j]|) for j≠i
    """
    n = A.shape[0]
    diag_elements = np.abs(np.diag(A))
    off_diag_sums = np.sum(np.abs(A), axis=1) - diag_elements
    
    dominance_ratios = diag_elements / off_diag_sums
    min_ratio = np.min(dominance_ratios)
    
    is_dominant = np.all(dominance_ratios >= 1.0)
    
    return is_dominant, min_ratio

def gauss_seidel_solve(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None, 
                      max_iter: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, int, list]:
    """
    Solve Ax = b using Gauss-Seidel iteration.
    
    Returns:
        x: Solution vector
        iterations: Number of iterations performed
        residual_history: List of residual norms at each iteration
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    residual_history = []
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_old[i+1:])) / A[i, i]
        
        # Compute residual
        residual = np.linalg.norm(A @ x - b)
        residual_history.append(residual)
        
        # Check convergence
        if residual < tol:
            return x, iteration + 1, residual_history
    
    print(f"Warning: Gauss-Seidel did not converge in {max_iter} iterations")
    return x, max_iter, residual_history

def jacobi_solve(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
                max_iter: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, int, list]:
    """
    Solve Ax = b using Jacobi iteration.
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    residual_history = []
    
    D_inv = 1.0 / np.diag(A)
    
    for iteration in range(max_iter):
        x_new = D_inv * (b - (A @ x) + np.diag(A) * x)
        
        # Compute residual
        residual = np.linalg.norm(A @ x_new - b)
        residual_history.append(residual)
        
        # Check convergence
        if residual < tol:
            return x_new, iteration + 1, residual_history
        
        x = x_new
    
    print(f"Warning: Jacobi did not converge in {max_iter} iterations")
    return x, max_iter, residual_history

def test_iterative_methods(A: np.ndarray, b: np.ndarray, x_true: np.ndarray) -> Dict[str, Any]:
    """Test iterative solution methods."""
    print(f"\n{'='*60}")
    print("STRATEGY 4: ITERATIVE METHODS")
    print(f"{'='*60}")
    
    # Check diagonal dominance
    is_dominant, min_ratio = check_diagonal_dominance(A)
    
    print("""
    ITERATIVE METHODS THEORY:
    
    1. CONVERGENCE REQUIREMENT: For guaranteed convergence, the matrix should be 
       diagonally dominant or have spectral radius < 1 for the iteration matrix.
    
    2. GAUSS-SEIDEL: Uses updated values immediately, often converges faster than Jacobi
       but requires sequential computation.
    
    3. JACOBI: Uses values from previous iteration, easily parallelizable but may 
       converge slower.
    
    4. ILL-CONDITIONED CHALLENGES: Poor conditioning can lead to very slow convergence 
       or divergence, making iterative methods less suitable than direct methods.
    """)
    
    print(f"Diagonal dominance check:")
    print(f"  Is diagonally dominant: {is_dominant}")
    print(f"  Minimum dominance ratio: {min_ratio:.2f}")
    
    if min_ratio < 0.1:
        print(f"  WARNING: Very poor diagonal dominance - iterative methods may not converge!")
    
    # Test Gauss-Seidel
    print(f"\nTesting Gauss-Seidel iteration...")
    start_time = time.time()
    x_gs, iter_gs, residuals_gs = gauss_seidel_solve(A, b, max_iter=2000, tol=1e-12)
    gs_time = time.time() - start_time
    
    result_gs = evaluate_solution(A, b, x_gs, x_true, f"Gauss-Seidel ({iter_gs} iterations)")
    result_gs['iterations'] = iter_gs
    result_gs['computation_time'] = gs_time
    result_gs['final_residual'] = residuals_gs[-1] if residuals_gs else float('inf')
    
    # Test Jacobi
    print(f"\nTesting Jacobi iteration...")
    start_time = time.time()
    x_jac, iter_jac, residuals_jac = jacobi_solve(A, b, max_iter=2000, tol=1e-12)
    jac_time = time.time() - start_time
    
    result_jac = evaluate_solution(A, b, x_jac, x_true, f"Jacobi ({iter_jac} iterations)")
    result_jac['iterations'] = iter_jac
    result_jac['computation_time'] = jac_time
    result_jac['final_residual'] = residuals_jac[-1] if residuals_jac else float('inf')
    
    # Test with preconditioning
    print(f"\nTesting preconditioned iterative methods...")
    A_precond, b_precond, D_inv = precondition_jacobi(A, b)
    
    x_gs_precond, iter_gs_precond, _ = gauss_seidel_solve(A_precond, b_precond, max_iter=2000, tol=1e-12)
    result_gs_precond = evaluate_solution(A, b, x_gs_precond, x_true, f"Preconditioned Gauss-Seidel ({iter_gs_precond} iter)")
    
    return {
        'diagonal_dominance': {'is_dominant': is_dominant, 'min_ratio': min_ratio},
        'gauss_seidel': result_gs,
        'jacobi': result_jac,
        'gauss_seidel_preconditioned': result_gs_precond
    }

# ============================================================================
# COMPREHENSIVE COMPARISON AND ANALYSIS
# ============================================================================

def run_comprehensive_analysis():
    """Run complete analysis comparing all methods."""
    print("🔬 COMPREHENSIVE ANALYSIS: ILL-CONDITIONED LINEAR SYSTEMS")
    print("=" * 80)
    
    # Create test system
    A, b, x_true = create_ill_conditioned_system(n=9, condition_target=1e12)
    print_system_info(A, b, "Test System")
    
    # Store all results
    results = {}
    
    # Test all strategies
    results['partial_pivoting'] = test_partial_pivoting(A, b, x_true)
    results['conditioning'] = test_matrix_conditioning(A, b, x_true)
    results['higher_precision'] = test_higher_precision(A, b, x_true)
    results['iterative'] = test_iterative_methods(A, b, x_true)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Method':<30} {'Rel. Error':<12} {'Residual':<12} {'Time (s)':<10}")
    print("-" * 70)
    
    # Partial pivoting
    pp_result = results['partial_pivoting']
    print(f"{'Partial Pivoting':<30} {pp_result['relative_error']:<12.2e} {pp_result['residual_norm']:<12.2e} {pp_result.get('computation_time', 0):<10.4f}")
    
    # Conditioning methods
    eq_result = results['conditioning']['equilibrated']
    jac_result = results['conditioning']['jacobi']
    print(f"{'Matrix Equilibration':<30} {eq_result['relative_error']:<12.2e} {eq_result['residual_norm']:<12.2e} {'N/A':<10}")
    print(f"{'Jacobi Preconditioning':<30} {jac_result['relative_error']:<12.2e} {jac_result['residual_norm']:<12.2e} {'N/A':<10}")
    
    # Higher precision
    hp_result = results['higher_precision']
    f128_result = hp_result['float128']
    print(f"{'NumPy float128':<30} {f128_result['relative_error']:<12.2e} {f128_result['residual_norm']:<12.2e} {f128_result.get('computation_time', 0):<10.4f}")
    
    if hp_result['mpmath'] is not None:
        mp_result = hp_result['mpmath']
        print(f"{'mpmath (100-bit)':<30} {mp_result['relative_error']:<12.2e} {mp_result['residual_norm']:<12.2e} {mp_result.get('computation_time', 0):<10.4f}")
    
    # Iterative methods
    iter_results = results['iterative']
    gs_result = iter_results['gauss_seidel']
    jac_iter_result = iter_results['jacobi']
    print(f"{'Gauss-Seidel':<30} {gs_result['relative_error']:<12.2e} {gs_result['residual_norm']:<12.2e} {gs_result.get('computation_time', 0):<10.4f}")
    print(f"{'Jacobi':<30} {jac_iter_result['relative_error']:<12.2e} {jac_iter_result['residual_norm']:<12.2e} {jac_iter_result.get('computation_time', 0):<10.4f}")
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR ILL-CONDITIONED SYSTEMS")
    print(f"{'='*80}")
    
    condition_number = np.linalg.cond(A)
    
    print(f"""
📊 ANALYSIS SUMMARY:

System Characteristics:
• Condition number: {condition_number:.2e}
• Severity: {'EXTREMELY' if condition_number > 1e10 else 'MODERATELY'} ill-conditioned
• Matrix element range: [{np.min(A):.2e}, {np.max(A):.2e}]

🎯 RECOMMENDED APPROACH:

For your specific case (condition number ~10¹²):

1. 🥇 PRIMARY RECOMMENDATION: Higher-Precision Arithmetic
   • Use numpy.float128 for 4x improvement in precision
   • Consider mpmath for critical applications requiring maximum accuracy
   • Provides best accuracy-to-effort ratio for severely ill-conditioned systems

2. 🥈 SECONDARY RECOMMENDATION: Matrix Equilibration + Partial Pivoting
   • Equilibrate matrix to improve condition number by ~{np.linalg.cond(A)/results['conditioning']['condition_improvements']['equilibrated']:.1f}x
   • Always use partial pivoting for numerical stability
   • Good balance of accuracy improvement and computational efficiency

3. ⚠️  AVOID: Pure Iterative Methods
   • Diagonal dominance ratio: {iter_results['diagonal_dominance']['min_ratio']:.2f}
   • Convergence {'POOR' if iter_results['diagonal_dominance']['min_ratio'] < 1 else 'ACCEPTABLE'}
   • Not suitable for this level of ill-conditioning without preconditioning

💡 PRACTICAL STRATEGY:
1. Start with matrix equilibration to improve conditioning
2. Use numpy.float128 for the actual solve
3. Always verify solution with residual check
4. Consider regularization if physical meaning allows

⚡ COMPUTATIONAL EFFICIENCY:
• float128: ~{f128_result.get('computation_time', 0)/pp_result.get('computation_time', 1):.1f}x slower than double precision
• mpmath: Much slower but most accurate
• Equilibration: Minimal overhead, significant benefit
    """)

if __name__ == "__main__":
    run_comprehensive_analysis() 