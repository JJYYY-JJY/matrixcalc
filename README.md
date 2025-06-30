# Gaussian Elimination Solver

A complete Python implementation for solving systems of linear equations **Ax = b** using Gaussian elimination with partial pivoting. **Now supports rectangular matrices of any size!**

## Overview

This project implements the Gaussian elimination algorithm, a fundamental method in numerical linear algebra for solving systems of linear equations. The implementation includes **partial pivoting** for enhanced numerical stability and proper handling of all types of linear systems including overdetermined and underdetermined cases.

### Key Features

- **Rectangular Matrix Support**: Works with any matrix size (n×m), not just square matrices
- **Gaussian Elimination with Partial Pivoting**: Ensures numerical stability by selecting the largest pivot element
- **Complete System Analysis**: Handles square, overdetermined, and underdetermined systems
- **Comprehensive Error Handling**: Detects and handles cases of no solution or infinite solutions
- **Interactive Interface**: Menu-driven command-line interface for easy use
- **Clean API**: Simple `solve(A, b)` function interface
- **Extensive Testing**: Unit tests covering all edge cases and matrix types
- **Educational Content**: Comprehensive explanations of different system types

## Mathematical Background

**Gaussian Elimination** is a systematic method for solving linear systems by transforming the augmented matrix `[A|b]` into row echelon form through elementary row operations. This implementation now supports **rectangular matrices** of any size:

### System Types Supported

1. **Square Systems (n = m)**: Traditional case with equal equations and unknowns
   - May have unique solution, no solution, or infinite solutions
   
2. **Overdetermined Systems (n > m)**: More equations than unknowns
   - Common in data fitting and least squares problems
   - May have no solution (inconsistent) or unique solution (consistent)
   
3. **Underdetermined Systems (n < m)**: More unknowns than equations
   - Typically have infinite solutions (free variables)
   - May have no solution if inconsistent

### Algorithm Steps

1. **Forward Elimination**: Transform the matrix to row echelon form
   - Performs min(n,m) elimination steps for rectangular matrices
   - Uses row operations to eliminate elements below pivots
2. **Partial Pivoting**: At each step, swap rows to place the largest absolute value element on the diagonal
3. **Solution Analysis**: Check rank conditions to determine solution type
4. **Back Substitution**: Solve for variables when unique solution exists

**Partial Pivoting** is crucial for:
- Avoiding division by zero when diagonal elements are zero
- Minimizing numerical errors and improving stability
- Handling ill-conditioned matrices more robustly

## Installation

### Option 1: Development Installation (Recommended)

1. Clone or download this repository
2. Install the package in development mode:

```bash
pip install -e .
```

This will install all dependencies and make the package importable from anywhere.

### Option 2: Manual Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Note: If using this method, run tests with `python -m pytest` instead of just `pytest`.

## Usage

### Interactive Interface (Recommended)

For the best user experience, use the interactive command-line interface:

```bash
python interactive.py
```

This provides a menu-driven interface where you can:
- 📝 **Enter your own matrices and vectors** with guided input
- 📚 **Choose from pre-defined examples** covering different scenarios
- 📖 **Learn about Gaussian elimination** with educational content
- 🎬 **Run comprehensive demonstrations** with detailed explanations
- ✅ **See step-by-step solutions** with verification

### Basic Programming Examples

**Square System:**
```python
import numpy as np
from gaussian_elimination import solve

# Define system: 2x + 3y = 7, x - y = 1
A = np.array([[2, 3], [1, -1]], dtype=float)
b = np.array([7, 1], dtype=float)

try:
    solution = solve(A, b)
    print(f"Solution: x = {solution}")  # Output: [2. 1.]
except ValueError as e:
    print(f"Error: {e}")
```

**Overdetermined System (3 equations, 2 unknowns):**
```python
# System that is consistent and has a unique solution
A = np.array([[1, 1], [1, 2], [2, 3]], dtype=float)
b = np.array([3, 4, 7], dtype=float)

try:
    solution = solve(A, b)
    print(f"Solution: x = {solution}")  # Output: [2. 1.]
    print("All 3 equations are satisfied!")
except ValueError as e:
    print(f"Error: {e}")
```

**Underdetermined System (2 equations, 3 unknowns):**
```python
# System with infinite solutions
A = np.array([[1, 2, 3], [2, 4, 6]], dtype=float)
b = np.array([6, 12], dtype=float)

try:
    solution = solve(A, b)
    print(f"Unexpected unique solution: {solution}")
except ValueError as e:
    print(f"Expected: {e}")  # Will show "infinite solutions"
```

### Running the Examples

Execute the demonstration script for detailed examples:

```bash
python main.py
```

This will show examples of:
- Square systems with unique solutions
- Systems requiring partial pivoting
- Systems with no solution
- Systems with infinite solutions

The interactive interface includes additional examples for:
- Overdetermined systems (3×2, 4×2, etc.)
- Underdetermined systems (2×3, 2×4, etc.)
- Consistent and inconsistent rectangular systems

## Testing

Run the comprehensive test suite:

```bash
pytest
```

Or for verbose output:

```bash
pytest -v
```

## Project Structure

```
gaussian-elimination-repo/
├── gaussian_elimination/
│   ├── __init__.py          # Package initialization
│   ├── core.py              # Core algorithm implementation
│   └── utils.py             # Helper functions for matrix display
├── tests/
│   └── test_solver.py       # Comprehensive unit tests
├── interactive.py           # Interactive command-line interface
├── main.py                  # Demonstration script
├── requirements.txt         # Project dependencies
├── README.md               # This file
└── .gitignore              # Git ignore patterns
```

## Algorithm Details

The implementation follows these steps:

1. **Augmentation**: Combine matrix A and vector b into `[A|b]`
2. **Forward Elimination with Partial Pivoting**:
   - For each column, find the row with the largest absolute value in that column
   - Swap rows to bring the best pivot to the diagonal
   - Eliminate all elements below the pivot
3. **Solution Detection**:
   - Check for inconsistent systems (no solution)
   - Check for underdetermined systems (infinite solutions)
4. **Back Substitution**: Calculate the unique solution if it exists

## Error Handling

The solver raises `ValueError` exceptions for:
- **No Solution**: When the system is inconsistent
- **Infinite Solutions**: When the system is underdetermined
- **Invalid Input**: When matrix dimensions don't match

## Dependencies

- **NumPy**: For efficient matrix operations and numerical computing
- **pytest**: For running the test suite

## Contributing

This project is designed for educational purposes. Feel free to extend it with additional features such as:
- LU decomposition
- Matrix inversion
- Determinant calculation
- Support for complex numbers

## License

This project is available for educational and research purposes. 