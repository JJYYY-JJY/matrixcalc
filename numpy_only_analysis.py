#!/usr/bin/env python3
"""
Numerical Strategies for Ill-Conditioned Systems (NumPy-only version)

This demonstrates the key approaches for solving severely ill-conditioned linear systems
using only numpy, making it compatible with any Python environment.
"""

import numpy as np
import time

def create_sample_ill_conditioned_system():
    """Create a sample 9×9 ill-conditioned system similar to the user's problem."""
    np.random.seed(42)
    A = np.random.randn(9, 9)
    
    # Make severely ill-conditioned with varying scales (1 to 10^12)
    scales = np.logspace(0, 12, 9)
    A = A * scales[np.newaxis, :]
    
    # Add structure to emphasize diagonal
    for i in range(9):
        for j in range(9):
            if abs(i - j) <= 1:
                A[i, j] *= 10
    
    x_true = np.ones(9)
    b = A @ x_true
    
    return A, b, x_true

def analyze_system(A, b):
    """Analyze the linear system characteristics."""
    condition_number = np.linalg.cond(A)
    matrix_min = np.min(A)
    matrix_max = np.max(A)
    
    print(f"System Analysis:")
    print(f"  Size: {A.shape[0]}×{A.shape[1]}")
    print(f"  Condition number: {condition_number:.2e}")
    print(f"  Matrix elements range: [{matrix_min:.2e}, {matrix_max:.2e}]")
    print(f"  Magnitude span: {matrix_max/abs(matrix_min):.2e}")
    print(f"  Severity: {'EXTREMELY' if condition_number > 1e10 else 'MODERATELY'} ill-conditioned")
    return condition_number

def evaluate_solution(A, b, x, x_true, method_name):
    """Evaluate solution quality."""
    residual = A @ x - b
    error = x - x_true
    
    residual_norm = np.linalg.norm(residual)
    error_norm = np.linalg.norm(error)
    relative_error = error_norm / np.linalg.norm(x_true)
    max_residual = np.max(np.abs(residual))
    
    print(f"\n{method_name} Results:")
    print(f"  Residual norm ||Ax - b||: {residual_norm:.2e}")
    print(f"  Error norm ||x - x_true||: {error_norm:.2e}")
    print(f"  Relative error: {relative_error:.2e}")
    print(f"  Max residual component: {max_residual:.2e}")
    
    if residual_norm < 1e-10:
        quality = "EXCELLENT"
    elif residual_norm < 1e-6:
        quality = "GOOD"
    elif residual_norm < 1e-3:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"
    
    print(f"  Quality: {quality}")
    
    return {
        'residual_norm': residual_norm,
        'error_norm': error_norm,
        'relative_error': relative_error,
        'quality': quality
    }

def strategy_1_partial_pivoting(A, b, x_true):
    """Strategy 1: Gaussian Elimination with Partial Pivoting"""
    print(f"\n{'='*70}")
    print("STRATEGY 1: GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING")
    print(f"{'='*70}")
    
    print("""
WHY PARTIAL PIVOTING IS ESSENTIAL FOR ILL-CONDITIONED SYSTEMS:

1. NUMERICAL STABILITY: Small pivots amplify rounding errors exponentially.
   Partial pivoting selects the largest available pivot to minimize this effect.

2. ERROR CONTROL: Prevents division by very small numbers that can cause
   catastrophic loss of precision in subsequent calculations.

3. ROBUSTNESS: Handles near-singular matrices more gracefully than naive
   Gaussian elimination.

4. STANDARD PRACTICE: All professional linear algebra libraries (LAPACK,
   used by NumPy) implement partial pivoting by default.
    """)
    
    start_time = time.time()
    x_pivot = np.linalg.solve(A, b)  # NumPy uses LAPACK with partial pivoting
    pivot_time = time.time() - start_time
    
    result = evaluate_solution(A, b, x_pivot, x_true, "NumPy (LAPACK with Partial Pivoting)")
    print(f"  Computation time: {pivot_time:.4f} seconds")
    
    return result

def equilibrate_matrix(A, b):
    """Equilibrate matrix to improve condition number."""
    # Compute row and column scaling factors using 2-norm
    row_norms = np.sqrt(np.sum(A**2, axis=1))
    col_norms = np.sqrt(np.sum(A**2, axis=0))
    
    # Avoid division by zero
    row_scales = np.where(row_norms > 1e-14, 1.0 / row_norms, 1.0)
    col_scales = np.where(col_norms > 1e-14, 1.0 / col_norms, 1.0)
    
    # Apply scaling: D1 * A * D2
    A_equilibrated = np.diag(row_scales) @ A @ np.diag(col_scales)
    b_equilibrated = row_scales * b
    
    return A_equilibrated, b_equilibrated, row_scales, col_scales

def strategy_2_matrix_conditioning(A, b, x_true):
    """Strategy 2: Matrix Preprocessing/Conditioning"""
    print(f"\n{'='*70}")
    print("STRATEGY 2: MATRIX PREPROCESSING/CONDITIONING")
    print(f"{'='*70}")
    
    print("""
MATRIX EQUILIBRATION THEORY:

1. PURPOSE: Reduce condition number by balancing row and column scales
2. METHOD: Apply diagonal scaling matrices D₁ and D₂ such that
   D₁AD₂ has more uniform element magnitudes
3. MATHEMATICAL BASIS: If D₁AD₂y = D₁b, then x = D₂y is the solution to Ax = b
4. BENEFIT: Can reduce condition number by orders of magnitude

IMPLEMENTATION:
- Row scaling: D₁ = diag(1/||A[i,:]||₂)
- Column scaling: D₂ = diag(1/||A[:,j]||₂)
    """)
    
    original_cond = np.linalg.cond(A)
    print(f"Original condition number: {original_cond:.2e}")
    
    # Apply equilibration
    A_eq, b_eq, row_scales, col_scales = equilibrate_matrix(A, b)
    equilibrated_cond = np.linalg.cond(A_eq)
    
    improvement = original_cond / equilibrated_cond
    print(f"Equilibrated condition number: {equilibrated_cond:.2e}")
    print(f"Improvement factor: {improvement:.1f}x")
    
    # Solve equilibrated system
    y_eq = np.linalg.solve(A_eq, b_eq)
    x_eq = col_scales * y_eq  # Recover original solution
    
    result = evaluate_solution(A, b, x_eq, x_true, "Matrix Equilibration")
    
    # Show effect on matrix element distribution
    print(f"\nMatrix element distribution:")
    print(f"  Original: min={np.min(A):.2e}, max={np.max(A):.2e}, span={np.max(A)/abs(np.min(A)):.2e}")
    print(f"  Equilibrated: min={np.min(A_eq):.2e}, max={np.max(A_eq):.2e}, span={np.max(A_eq)/abs(np.min(A_eq)):.2e}")
    
    return result, improvement

def strategy_3_higher_precision(A, b, x_true):
    """Strategy 3: Higher-Precision Arithmetic"""
    print(f"\n{'='*70}")
    print("STRATEGY 3: HIGHER-PRECISION ARITHMETIC")
    print(f"{'='*70}")
    
    print("""
PRECISION ARITHMETIC BENEFITS:

1. EXTENDED PRECISION (float128): 
   - Standard float64: ~15-16 significant decimal digits
   - Extended float128: ~18-19 significant decimal digits
   - Reduces intermediate rounding errors by ~1000x

2. WHY IT HELPS ILL-CONDITIONED SYSTEMS:
   - Condition number amplifies input errors by factor of κ(A)
   - With κ(A) ≈ 10¹², float64 precision (10⁻¹⁵) becomes 10⁻³ in output
   - float128 precision (10⁻¹⁸) becomes 10⁻⁶ in output - much more acceptable

3. TRADE-OFFS:
   - Computational cost: ~2-4x slower than float64
   - Memory usage: 2x more memory required
   - Platform dependent: Not all systems support float128
    """)
    
    # Check if float128 is available
    try:
        test_array = np.array([1.0], dtype=np.float128)
        float128_available = True
        print("✅ float128 is available on this system")
    except:
        float128_available = False
        print("❌ float128 is not available on this system")
        return None
    
    if float128_available:
        # Convert to higher precision
        A_hp = A.astype(np.float128)
        b_hp = b.astype(np.float128)
        
        start_time = time.time()
        x_hp = np.linalg.solve(A_hp, b_hp)
        hp_time = time.time() - start_time
        
        # Convert back to float64 for comparison
        x_result = x_hp.astype(np.float64)
        
        result = evaluate_solution(A, b, x_result, x_true, "NumPy float128 (Extended Precision)")
        print(f"  Computation time: {hp_time:.4f} seconds")
        
        return result
    
    return None

def strategy_4_iterative_methods(A, b, x_true):
    """Strategy 4: Iterative Methods"""
    print(f"\n{'='*70}")
    print("STRATEGY 4: ITERATIVE METHODS")
    print(f"{'='*70}")
    
    print("""
ITERATIVE METHODS FOR ILL-CONDITIONED SYSTEMS:

1. CONVERGENCE THEORY:
   - Convergence depends on spectral radius of iteration matrix
   - For Jacobi: ρ(D⁻¹(L+U)) < 1 required for convergence
   - For Gauss-Seidel: ρ((D+L)⁻¹U) < 1 required
   - Ill-conditioning often violates these conditions

2. DIAGONAL DOMINANCE:
   - Sufficient condition: |A[i,i]| > Σ|A[i,j]| for j≠i
   - Ensures convergence but is often violated in practice

3. WHY ITERATIVE METHODS STRUGGLE WITH ILL-CONDITIONING:
   - Slow convergence due to large condition number
   - May not converge at all for severely ill-conditioned systems
   - Round-off errors can prevent convergence to desired accuracy
    """)
    
    # Check diagonal dominance
    diag_elements = np.abs(np.diag(A))
    off_diag_sums = np.sum(np.abs(A), axis=1) - diag_elements
    dominance_ratios = diag_elements / off_diag_sums
    min_ratio = np.min(dominance_ratios)
    is_dominant = np.all(dominance_ratios >= 1.0)
    
    print(f"Diagonal dominance analysis:")
    print(f"  Is diagonally dominant: {is_dominant}")
    print(f"  Minimum dominance ratio: {min_ratio:.3f}")
    print(f"  Required for convergence: ratio ≥ 1.0")
    
    if min_ratio < 0.1:
        print(f"  ⚠️  WARNING: Very poor diagonal dominance - convergence unlikely!")
    
    # Simple Jacobi iteration
    def jacobi_iteration(A, b, max_iter=1000, tol=1e-10):
        n = len(b)
        x = np.zeros(n)
        D_inv = 1.0 / np.diag(A)
        
        for i in range(max_iter):
            x_new = D_inv * (b - (A @ x) + np.diag(A) * x)
            
            residual = np.linalg.norm(A @ x_new - b)
            if residual < tol:
                return x_new, i + 1, residual
            
            x = x_new
        
        return x, max_iter, residual
    
    print(f"\nTesting Jacobi iteration...")
    start_time = time.time()
    x_jacobi, iterations, final_residual = jacobi_iteration(A, b, max_iter=2000)
    jacobi_time = time.time() - start_time
    
    print(f"  Iterations: {iterations}")
    print(f"  Final residual: {final_residual:.2e}")
    print(f"  Computation time: {jacobi_time:.4f} seconds")
    
    if iterations >= 2000:
        print(f"  ❌ Did not converge in {iterations} iterations")
        print(f"  This demonstrates why iterative methods struggle with ill-conditioning")
    else:
        print(f"  ✅ Converged in {iterations} iterations")
    
    result = evaluate_solution(A, b, x_jacobi, x_true, f"Jacobi Iteration ({iterations} iter)")
    
    return result

def comprehensive_comparison():
    """Run comprehensive comparison of all strategies."""
    print("🔬 COMPREHENSIVE ANALYSIS: ILL-CONDITIONED LINEAR SYSTEMS")
    print("=" * 80)
    print("Comparing numerical strategies for severely ill-conditioned 9×9 systems")
    print("with elements ranging from single digits to 10¹²")
    print("=" * 80)
    
    # Create test system
    A, b, x_true = create_sample_ill_conditioned_system()
    condition_number = analyze_system(A, b)
    
    print(f"\nTrue solution (for verification): {x_true}")
    
    # Test all strategies
    results = {}
    
    # Strategy 1: Partial Pivoting
    results['partial_pivoting'] = strategy_1_partial_pivoting(A, b, x_true)
    
    # Strategy 2: Matrix Conditioning
    results['conditioning'], improvement = strategy_2_matrix_conditioning(A, b, x_true)
    
    # Strategy 3: Higher Precision
    hp_result = strategy_3_higher_precision(A, b, x_true)
    if hp_result:
        results['higher_precision'] = hp_result
    
    # Strategy 4: Iterative Methods
    results['iterative'] = strategy_4_iterative_methods(A, b, x_true)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Method':<30} {'Relative Error':<15} {'Residual Norm':<15} {'Quality':<12}")
    print("-" * 75)
    
    for method, result in results.items():
        if result:
            method_name = method.replace('_', ' ').title()
            print(f"{method_name:<30} {result['relative_error']:<15.2e} {result['residual_norm']:<15.2e} {result['quality']:<12}")
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("🎯 RECOMMENDATIONS FOR YOUR ILL-CONDITIONED SYSTEM")
    print(f"{'='*80}")
    
    print(f"""
📊 ANALYSIS FOR CONDITION NUMBER ~10¹²:

🥇 PRIMARY RECOMMENDATION: Matrix Equilibration + Higher Precision
   ✅ Equilibration improves condition number by ~{improvement:.1f}x
   ✅ float128 provides ~1000x better precision than float64
   ✅ Combined approach addresses both conditioning and precision issues
   ⚡ Moderate computational overhead (~2-4x slower)

🥈 SECONDARY RECOMMENDATION: Matrix Equilibration + Partial Pivoting  
   ✅ Significant improvement in condition number
   ✅ Uses standard double precision - widely compatible
   ✅ Minimal computational overhead
   ⚠️  May still have precision limitations for κ(A) > 10¹⁰

❌ NOT RECOMMENDED: Pure Iterative Methods
   ❌ Diagonal dominance ratio: {np.min(np.abs(np.diag(A)) / (np.sum(np.abs(A), axis=1) - np.abs(np.diag(A)))):.2f} << 1
   ❌ Convergence unlikely for this level of ill-conditioning
   ❌ Would require sophisticated preconditioning

💡 PRACTICAL IMPLEMENTATION STRATEGY:
1. Always equilibrate your matrix first
2. Use float128 if available and accuracy is critical
3. Verify solution quality by checking ||Ax - b||
4. Consider regularization if problem allows

⚗️  FOR YOUR SPECIFIC CASE:
   Given elements ranging from single digits to 10¹², equilibration is ESSENTIAL
   The ~{np.max(A)/abs(np.min(A)):.0e}x magnitude span makes the raw system nearly unsolvable
    """)

if __name__ == "__main__":
    comprehensive_comparison() 