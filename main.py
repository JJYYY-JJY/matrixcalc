#!/usr/bin/env python3
"""
Main demonstration script for the Gaussian elimination solver.

This script showcases the functionality of the gaussian_elimination package
with examples of different types of linear systems and their solutions.
"""

import numpy as np
from gaussian_elimination import solve, print_system, print_matrix, format_solution
from gaussian_elimination.utils import print_augmented_matrix


def print_separator(title: str) -> None:
    """Print a formatted separator with title."""
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demonstrate_unique_solution():
    """Demonstrate solving a system with a unique solution."""
    print_separator("EXAMPLE 1: SYSTEM WITH UNIQUE SOLUTION")
    
    # Define a 3x3 system with unique solution
    A = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    
    print("Consider the following system of linear equations:")
    print_system(A, b)
    
    print("Coefficient matrix A:")
    print_matrix(A, precision=3)
    
    print("Right-hand side vector b:")
    print_matrix(b.reshape(1, -1), precision=3)
    
    print("Augmented matrix [A|b]:")
    print_augmented_matrix(A, b)
    
    try:
        print("Solving using Gaussian elimination with partial pivoting...")
        solution = solve(A, b)
        
        print("✅ SOLUTION FOUND!")
        print(f"Solution: {format_solution(solution)}")
        
        # Verify the solution
        print("\nVerification (Ax = b):")
        result = np.dot(A, solution)
        print(f"A × x = {result}")
        print(f"b     = {b}")
        print(f"Difference: {np.abs(result - b)}")
        
        if np.allclose(result, b, rtol=1e-10):
            print("✅ Verification successful! The solution is correct.")
        else:
            print("❌ Verification failed!")
            
    except ValueError as e:
        print(f"❌ Error: {e}")
    
    print()


def demonstrate_no_solution():
    """Demonstrate a system with no solution."""
    print_separator("EXAMPLE 2: SYSTEM WITH NO SOLUTION")
    
    # Define an inconsistent system
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    b = np.array([1, 2, 4], dtype=float)  # Inconsistent with the rank of A
    
    print("Consider the following system (inconsistent):")
    print_system(A, b)
    
    print("This system is inconsistent because the coefficient matrix has")
    print("linearly dependent rows, but the augmented matrix has a different rank.")
    print()
    
    print("Coefficient matrix A:")
    print_matrix(A, precision=3)
    
    print("Right-hand side vector b:")
    print_matrix(b.reshape(1, -1), precision=3)
    
    try:
        print("Attempting to solve...")
        solution = solve(A, b)
        print(f"Unexpected solution: {solution}")
        
    except ValueError as e:
        print(f"✅ Expected error caught: {e}")
        print("This confirms that the system has no solution.")
    
    print()


def demonstrate_infinite_solutions():
    """Demonstrate a system with infinite solutions."""
    print_separator("EXAMPLE 3: SYSTEM WITH INFINITE SOLUTIONS")
    
    # Define a system with dependent equations
    A = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]
    ], dtype=float)
    b = np.array([6, 12, 3], dtype=float)
    
    print("Consider the following system (dependent equations):")
    print_system(A, b)
    
    print("Notice that the second equation is exactly twice the first equation.")
    print("This means the system has infinitely many solutions (underdetermined).")
    print()
    
    print("Coefficient matrix A:")
    print_matrix(A, precision=3)
    
    print("Right-hand side vector b:")
    print_matrix(b.reshape(1, -1), precision=3)
    
    try:
        print("Attempting to solve...")
        solution = solve(A, b)
        print(f"Unexpected solution: {solution}")
        
    except ValueError as e:
        print(f"✅ Expected error caught: {e}")
        print("This confirms that the system has infinitely many solutions.")
    
    print()


def demonstrate_partial_pivoting():
    """Demonstrate the importance of partial pivoting."""
    print_separator("EXAMPLE 4: PARTIAL PIVOTING IN ACTION")
    
    # System that requires row swapping for numerical stability
    A = np.array([
        [0, 2, 3],
        [1, 1, 1],
        [2, 1, 1]
    ], dtype=float)
    b = np.array([13, 6, 5], dtype=float)
    
    print("Consider this system where the first pivot is zero:")
    print_system(A, b)
    
    print("Without partial pivoting, we would encounter division by zero.")
    print("Our algorithm automatically swaps rows to find the best pivot.")
    print()
    
    print("Original augmented matrix:")
    print_augmented_matrix(A, b)
    
    try:
        print("Solving with partial pivoting...")
        solution = solve(A, b)
        
        print("✅ SOLUTION FOUND!")
        print(f"Solution: {format_solution(solution)}")
        
        # Verify the solution
        print("\nVerification:")
        result = np.dot(A, solution)
        print(f"A × x = {result}")
        print(f"b     = {b}")
        
        if np.allclose(result, b, rtol=1e-10):
            print("✅ Solution verified!")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
    
    print()


def demonstrate_small_example():
    """Demonstrate with a simple 2x2 example."""
    print_separator("EXAMPLE 5: SIMPLE 2×2 SYSTEM")
    
    A = np.array([
        [2, 3],
        [1, -1]
    ], dtype=float)
    b = np.array([7, 1], dtype=float)
    
    print("A simple 2×2 system:")
    print_system(A, b)
    
    try:
        solution = solve(A, b)
        print("✅ SOLUTION FOUND!")
        print(f"Solution: {format_solution(solution)}")
        
        # Show step by step verification
        print("\nStep-by-step verification:")
        x1, x2 = solution
        print(f"Equation 1: 2×({x1:.6f}) + 3×({x2:.6f}) = {2*x1 + 3*x2:.6f} (should equal {b[0]})")
        print(f"Equation 2: 1×({x1:.6f}) - 1×({x2:.6f}) = {1*x1 - 1*x2:.6f} (should equal {b[1]})")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
    
    print()


def main():
    """Main function to run all demonstrations."""
    print("🔢 GAUSSIAN ELIMINATION SOLVER DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates the capabilities of our Gaussian elimination")
    print("solver with partial pivoting for different types of linear systems.")
    print("=" * 60)
    print()
    
    # Run all demonstrations
    demonstrate_small_example()
    demonstrate_unique_solution()
    demonstrate_partial_pivoting()
    demonstrate_no_solution()
    demonstrate_infinite_solutions()
    
    # Final summary
    print_separator("SUMMARY")
    print("✅ The Gaussian elimination solver successfully:")
    print("   • Solves systems with unique solutions using partial pivoting")
    print("   • Detects and reports systems with no solution")
    print("   • Detects and reports systems with infinite solutions")
    print("   • Handles numerical edge cases robustly")
    print()
    print("📖 For more information, see the README.md file")
    print("🧪 Run 'pytest' to execute the comprehensive test suite")
    print("=" * 60)


if __name__ == "__main__":
    main() 