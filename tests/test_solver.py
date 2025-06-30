"""
Unit tests for the Gaussian elimination solver.

This module contains comprehensive tests for all functionality of the
gaussian_elimination package, covering edge cases and different solution types.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from gaussian_elimination import solve
from gaussian_elimination.core import get_solution_info


class TestGaussianElimination:
    """Test cases for the main solve function."""
    
    def test_unique_solution_2x2(self):
        """Test a simple 2x2 system with unique solution."""
        # System: 2x + y = 3, x + y = 2
        # Solution: x = 1, y = 1
        A = np.array([[2, 1], [1, 1]], dtype=float)
        b = np.array([3, 2], dtype=float)
        
        solution = solve(A, b)
        expected = np.array([1, 1], dtype=float)
        
        assert_allclose(solution, expected, rtol=1e-10)
    
    def test_unique_solution_3x3(self):
        """Test a 3x3 system with unique solution."""
        # System from README example
        A = np.array([
            [2, 1, -1],
            [-3, -1, 2],
            [-2, 1, 2]
        ], dtype=float)
        b = np.array([8, -11, -3], dtype=float)
        
        solution = solve(A, b)
        
        # Verify solution by substitution
        result = np.dot(A, solution)
        assert_allclose(result, b, rtol=1e-10)
    
    def test_partial_pivoting_required(self):
        """Test a system that requires partial pivoting to solve."""
        # System where first element is zero
        A = np.array([
            [0, 1, 2],
            [1, 2, 1],
            [2, 1, 1]
        ], dtype=float)
        b = np.array([3, 4, 3], dtype=float)
        
        solution = solve(A, b)
        
        # Verify solution by substitution
        result = np.dot(A, solution)
        assert_allclose(result, b, rtol=1e-10)
    
    def test_numerical_stability(self):
        """Test numerical stability with small pivot elements."""
        # System with small numbers that could cause numerical issues
        A = np.array([
            [1e-10, 1],
            [1, 1]
        ], dtype=float)
        b = np.array([1, 2], dtype=float)
        
        solution = solve(A, b)
        
        # Verify solution by substitution
        result = np.dot(A, solution)
        assert_allclose(result, b, rtol=1e-8)
    
    def test_no_solution_inconsistent(self):
        """Test detection of inconsistent system (no solution)."""
        # System: x + y = 1, x + y = 2 (contradictory)
        A = np.array([
            [1, 1],
            [1, 1]
        ], dtype=float)
        b = np.array([1, 2], dtype=float)
        
        with pytest.raises(ValueError, match="no solution"):
            solve(A, b)
    
    def test_infinite_solutions_dependent_equations(self):
        """Test detection of dependent equations (infinite solutions)."""
        # System: x + y = 1, 2x + 2y = 2 (dependent equations)
        A = np.array([
            [1, 1],
            [2, 2]
        ], dtype=float)
        b = np.array([1, 2], dtype=float)
        
        with pytest.raises(ValueError, match="infinite solutions"):
            solve(A, b)
    
    def test_infinite_solutions_underdetermined(self):
        """Test detection of underdetermined system."""
        # System with zero row
        A = np.array([
            [1, 2],
            [0, 0]
        ], dtype=float)
        b = np.array([3, 0], dtype=float)
        
        with pytest.raises(ValueError, match="infinite solutions"):
            solve(A, b)
    
    def test_singular_matrix(self):
        """Test handling of singular matrix."""
        # Singular matrix (rank deficient)
        A = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [1, 1, 1]
        ], dtype=float)
        b = np.array([1, 2, 1], dtype=float)
        
        with pytest.raises(ValueError, match="infinite solutions"):
            solve(A, b)


class TestInputValidation:
    """Test cases for input validation."""
    
    def test_invalid_matrix_dimension(self):
        """Test rejection of non-2D matrix."""
        A = np.array([1, 2, 3])  # 1D array
        b = np.array([1, 2])
        
        with pytest.raises(ValueError, match="2-dimensional"):
            solve(A, b)
    
    def test_invalid_vector_dimension(self):
        """Test rejection of non-1D vector."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[1], [2]])  # 2D array
        
        with pytest.raises(ValueError, match="1-dimensional"):
            solve(A, b)
    
    def test_incompatible_dimensions(self):
        """Test rejection of incompatible matrix and vector dimensions."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)  # Wrong length
        
        with pytest.raises(ValueError, match="Incompatible dimensions"):
            solve(A, b)
    
    def test_non_square_matrix(self):
        """Test rejection of non-square matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)  # 2x3 matrix
        b = np.array([1, 2], dtype=float)
        
        with pytest.raises(ValueError, match="square"):
            solve(A, b)
    
    def test_empty_matrix(self):
        """Test handling of empty matrix."""
        A = np.array([], dtype=float).reshape(0, 0)
        b = np.array([], dtype=float)
        
        solution = solve(A, b)
        assert solution.shape == (0,)


class TestGetSolutionInfo:
    """Test cases for the get_solution_info utility function."""
    
    def test_unique_solution_info(self):
        """Test get_solution_info with unique solution."""
        A = np.array([[2, 1], [1, 1]], dtype=float)
        b = np.array([3, 2], dtype=float)
        
        solution_type, solution = get_solution_info(A, b)
        
        assert solution_type == "unique"
        assert_allclose(solution, [1, 1], rtol=1e-10)
    
    def test_no_solution_info(self):
        """Test get_solution_info with no solution."""
        A = np.array([[1, 1], [1, 1]], dtype=float)
        b = np.array([1, 2], dtype=float)
        
        solution_type, solution = get_solution_info(A, b)
        
        assert solution_type == "no_solution"
        assert solution.size == 0
    
    def test_infinite_solutions_info(self):
        """Test get_solution_info with infinite solutions."""
        A = np.array([[1, 1], [2, 2]], dtype=float)
        b = np.array([1, 2], dtype=float)
        
        solution_type, solution = get_solution_info(A, b)
        
        assert solution_type == "infinite_solutions"
        assert solution.size == 0


class TestNumericalAccuracy:
    """Test cases for numerical accuracy and stability."""
    
    def test_large_numbers(self):
        """Test with large numbers."""
        A = np.array([
            [1e6, 2e6],
            [3e6, 4e6]
        ], dtype=float)
        b = np.array([5e6, 11e6], dtype=float)
        
        solution = solve(A, b)
        result = np.dot(A, solution)
        assert_allclose(result, b, rtol=1e-10)
    
    def test_small_numbers(self):
        """Test with small numbers."""
        A = np.array([
            [1e-6, 2e-6],
            [3e-6, 4e-6]
        ], dtype=float)
        b = np.array([5e-6, 11e-6], dtype=float)
        
        solution = solve(A, b)
        result = np.dot(A, solution)
        assert_allclose(result, b, rtol=1e-6)
    
    def test_mixed_scale_numbers(self):
        """Test with numbers of different scales."""
        A = np.array([
            [1e-3, 1e6],
            [1e6, 1e-3]
        ], dtype=float)
        b = np.array([1e3, 1e3], dtype=float)
        
        solution = solve(A, b)
        result = np.dot(A, solution)
        assert_allclose(result, b, rtol=1e-6)


class TestSpecialCases:
    """Test cases for special mathematical scenarios."""
    
    def test_identity_matrix(self):
        """Test with identity matrix."""
        A = np.eye(3)
        b = np.array([1, 2, 3], dtype=float)
        
        solution = solve(A, b)
        assert_allclose(solution, b, rtol=1e-10)
    
    def test_diagonal_matrix(self):
        """Test with diagonal matrix."""
        A = np.array([
            [2, 0, 0],
            [0, 3, 0],
            [0, 0, 4]
        ], dtype=float)
        b = np.array([4, 9, 12], dtype=float)
        
        solution = solve(A, b)
        expected = np.array([2, 3, 3], dtype=float)
        assert_allclose(solution, expected, rtol=1e-10)
    
    def test_upper_triangular(self):
        """Test with upper triangular matrix."""
        A = np.array([
            [1, 2, 3],
            [0, 4, 5],
            [0, 0, 6]
        ], dtype=float)
        b = np.array([14, 23, 18], dtype=float)
        
        solution = solve(A, b)
        result = np.dot(A, solution)
        assert_allclose(result, b, rtol=1e-10)
    
    def test_negative_coefficients(self):
        """Test with negative coefficients."""
        A = np.array([
            [-1, 2],
            [3, -4]
        ], dtype=float)
        b = np.array([1, 5], dtype=float)
        
        solution = solve(A, b)
        result = np.dot(A, solution)
        assert_allclose(result, b, rtol=1e-10)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"]) 