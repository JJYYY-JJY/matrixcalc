#!/usr/bin/env python3
"""
EXACT SOLUTION FOR LINEAR SYSTEMS - FINDS ALL SOLUTIONS

🔥 REPLACE YOUR MATRIX AND VECTOR HERE - VERY TOP OF FILE! 🔥
"""

# ============================================================================
# 📝 ENTER YOUR SYSTEM HERE - REPLACE WITH YOUR ACTUAL VALUES!
# ============================================================================

A = [
    [6, 5, 1, 0, -3],
    [3, -2, -1, 8, 12],
    [-7, 1, 3, 0, 11],
    [13, 2, 0, -2, -7]
]
b = [0, 0, 0, 0]

# A = [
#     [2, 1, 0, 0, 3, -5],
#     [0, 5, -1, 8, -1, 4],
#     [3, 11, -9, 1, 6, 0],
#     [7, 0, 5, 5, -3, 2]
# ]
# b = [7, 0, 13, 11]

# OPTIONS:
show_rref = True          # Set to False to skip matrix analysis
show_steps = True         # Set to True to see detailed row operations
auto_proceed = True      # Set to True to skip user prompts for inconsistent systems
precision = 100            # Increase for higher precision (if you have mpmath)

# ============================================================================
# 🛑 STOP HERE - DON'T MODIFY ANYTHING BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
# ============================================================================

import numpy as np
from fractions import Fraction
import sys

# Configuration constants - centralized for easy maintenance
class Config:
    ZERO_THRESHOLD = 1e-14
    FRACTION_THRESHOLD = 1e-10
    MAX_FRACTION_DENOMINATOR = 100
    MAX_FRACTION_RATIO = 20
    EXCELLENT_QUALITY = 1e-10
    GOOD_QUALITY = 1e-6
    ACCEPTABLE_QUALITY = 1e-3
    HIGH_CONDITION_THRESHOLD = 1e8
    DEFAULT_MPMATH_PRECISION = 25

def validate_input(A, b):
    """Validate input matrices and vectors with comprehensive error checking."""
    try:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Input must be numeric arrays: {e}")
    
    if A.ndim != 2:
        raise ValueError(f"Matrix A must be 2D, got {A.ndim}D")
    if b.ndim != 1:
        raise ValueError(f"Vector b must be 1D, got {b.ndim}D")
    
    m, n = A.shape
    if m == 0 or n == 0:
        raise ValueError(f"Matrix A cannot be empty, got shape {A.shape}")
    if len(b) != m:
        raise ValueError(f"Incompatible dimensions: A is {A.shape}, b is {b.shape}")
    
    # Check for invalid values
    if not np.all(np.isfinite(A)):
        raise ValueError("Matrix A contains non-finite values (inf or nan)")
    if not np.all(np.isfinite(b)):
        raise ValueError("Vector b contains non-finite values (inf or nan)")
    
    return A, b

def format_number(x, threshold=Config.FRACTION_THRESHOLD, use_fractions=True):
    """Format number for readable output - use fractions for clean rationals, decimals otherwise."""
    if abs(x) < threshold:
        return "0"
    
    # Try fraction representation for reasonable numbers
    if use_fractions and abs(x) < 1000:
        try:
            frac = Fraction(x).limit_denominator(Config.MAX_FRACTION_DENOMINATOR)
            frac_error = abs(float(frac) - x)
            if frac_error < Config.FRACTION_THRESHOLD:
                if frac.denominator == 1:
                    return str(frac.numerator)
                elif abs(frac.numerator) < abs(frac.denominator) * Config.MAX_FRACTION_RATIO:
                    return str(frac)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    
    # Fallback to decimal representation
    if abs(x) < 1e-3 or abs(x) > 1e6:
        return f"{x:.3e}"
    elif abs(x) < 1:
        return f"{x:.6f}".rstrip('0').rstrip('.')
    elif abs(x) < 1000:
        formatted = f"{x:.6f}".rstrip('0').rstrip('.')
        return formatted if '.' in formatted else formatted
    else:
        return f"{x:.1f}"

def format_fraction(frac):
    """Format a Fraction object for display."""
    if frac.denominator == 1:
        return str(frac.numerator)
    else:
        return str(frac)

def print_matrix_exact(matrix, var_names=None, title="Matrix"):
    """Print matrix in exact fraction format."""
    try:
        m, n = len(matrix), len(matrix[0])
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n-1)] + ["b"]
        
        if len(var_names) != n:
            var_names = [f"col{i+1}" for i in range(n)]
        
        # Find maximum width needed
        max_width = 8  # minimum width
        for i in range(m):
            for j in range(n):
                max_width = max(max_width, len(format_fraction(matrix[i][j])))
        for name in var_names:
            max_width = max(max_width, len(str(name)))
        
        # Print header
        header = "  ".join(f"{name:>{max_width}}" for name in var_names)
        print(f"    {header}")
        print("    " + "-" * len(header))
        
        # Print rows
        for i in range(m):
            row_str = "  ".join(f"{format_fraction(matrix[i][j]):>{max_width}}" for j in range(n))
            print(f"R{i+1}: {row_str}")
                
    except Exception as e:
        print(f"Error printing {title}: {e}")

def print_matrix(matrix, var_names=None, title="Matrix"):
    """Print matrix in a readable format with error handling."""
    try:
        m, n = matrix.shape
        if var_names is None:
            var_names = [f"x{i}" for i in range(n-1)] + ["b"]
        
        if len(var_names) != n:
            var_names = [f"col{i}" for i in range(n)]
        
        # Find maximum width needed (with bounds checking)
        max_width = 8  # minimum width
        try:
            for i in range(m):
                for j in range(n):
                    max_width = max(max_width, len(format_number(matrix[i,j])))
            for name in var_names:
                max_width = max(max_width, len(str(name)))
        except:
            max_width = 12  # fallback width
        
        # Print header
        header = "  ".join(f"{name:>{max_width}}" for name in var_names)
        print(f"    {header}")
        print("    " + "-" * len(header))
        
        # Print rows
        for i in range(m):
            try:
                row_str = "  ".join(f"{format_number(matrix[i,j]):>{max_width}}" for j in range(n))
                print(f"R{i+1}: {row_str}")
            except Exception as e:
                print(f"R{i+1}: [Error displaying row: {e}]")
                
    except Exception as e:
        print(f"Error printing {title}: {e}")

def exact_rref(A, b, show_steps=False):
    """Compute RREF using exact fraction arithmetic."""
    try:
        # Convert to fractions
        m, n = len(A), len(A[0])
        augmented = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append(Fraction(A[i][j]).limit_denominator())
            row.append(Fraction(b[i]).limit_denominator())
            augmented.append(row)
        
        n_cols = n + 1  # including b column
        pivot_cols = []
        current_row = 0
        step_count = 0
        
        if show_steps:
            step_count += 1
            print(f"\nStep {step_count}: Initial augmented matrix [A|b]")
            print_matrix_exact(augmented)
        
        # Forward elimination to row echelon form
        for col in range(n):  # Only consider A columns for pivots
            if current_row >= m:
                break
                
            # Find pivot row (first non-zero entry in column)
            pivot_row = None
            for row in range(current_row, m):
                if augmented[row][col] != 0:
                    pivot_row = row
                    break
            
            # Skip if column is all zeros
            if pivot_row is None:
                if show_steps:
                    print(f"\nSkipping column {col+1} - all entries are zero")
                continue
            
            # Swap rows if needed
            if pivot_row != current_row:
                step_count += 1
                if show_steps:
                    print(f"\nStep {step_count}: Swap R{current_row+1} ↔ R{pivot_row+1}")
                
                augmented[current_row], augmented[pivot_row] = augmented[pivot_row], augmented[current_row]
                
                if show_steps:
                    print_matrix_exact(augmented)
            
            pivot_cols.append(col)
            pivot_val = augmented[current_row][col]
            
            # Scale pivot row to make pivot = 1
            if pivot_val != 1:
                step_count += 1
                if show_steps:
                    print(f"\nStep {step_count}: Scale R{current_row+1} by {format_fraction(Fraction(1)/pivot_val)}")
                
                for j in range(n_cols):
                    augmented[current_row][j] = augmented[current_row][j] / pivot_val
                
                if show_steps:
                    print_matrix_exact(augmented)
            
            # Eliminate column
            elimination_performed = False
            for row in range(m):
                if row != current_row and augmented[row][col] != 0:
                    if not elimination_performed and show_steps:
                        step_count += 1
                        print(f"\nStep {step_count}: Eliminate column {col+1}")
                        elimination_performed = True
                    
                    multiplier = augmented[row][col]
                    if show_steps:
                        print(f"R{row+1} = R{row+1} - ({format_fraction(multiplier)}) × R{current_row+1}")
                    
                    for j in range(n_cols):
                        augmented[row][j] = augmented[row][j] - multiplier * augmented[current_row][j]
            
            if elimination_performed and show_steps:
                print_matrix_exact(augmented)
            
            current_row += 1
        
        if show_steps:
            step_count += 1
            print(f"\nStep {step_count}: Final RREF")
            print_matrix_exact(augmented)
            print(f"Steps: {step_count}, Rank: {len(pivot_cols)}, Pivots: {[f'x{i+1}' for i in pivot_cols]}")
            if len(pivot_cols) < n:
                print(f"Free variables: {n - len(pivot_cols)}")
        
        return augmented, pivot_cols
        
    except Exception as e:
        print(f"Error in exact RREF computation: {e}")
        return None, []

def find_general_solution(rref_matrix, pivot_cols, n_vars):
    """Find the general solution from RREF in exact form."""
    try:
        if rref_matrix is None:
            return None
        
        m = len(rref_matrix)
        n_cols = len(rref_matrix[0])
        
        # Check for inconsistency
        for i in range(m):
            # Check if we have a row like [0 0 ... 0 | non_zero]
            all_zero_vars = True
            for j in range(n_vars):
                if rref_matrix[i][j] != 0:
                    all_zero_vars = False
                    break
            if all_zero_vars and rref_matrix[i][n_vars] != 0:
                return {'inconsistent': True, 'impossible_equation': f"0 = {format_fraction(rref_matrix[i][n_vars])}"}
        
        # Find free variables
        free_vars = []
        for j in range(n_vars):
            if j not in pivot_cols:
                free_vars.append(j)
        
        # Build solution
        if len(free_vars) == 0:
            # Unique solution
            solution = [Fraction(0)] * n_vars
            for i, col in enumerate(pivot_cols):
                if i < len(rref_matrix):
                    solution[col] = rref_matrix[i][n_vars]
            
            return {
                'type': 'unique',
                'solution': solution,
                'free_vars': [],
                'pivot_cols': pivot_cols
            }
        else:
            # Infinite solutions - parametric form
            # particular_solution: set all free variables to 0
            particular_solution = [Fraction(0)] * n_vars
            
            # Find particular solution (set free variables to 0)
            for i, col in enumerate(pivot_cols):
                if i < len(rref_matrix):
                    particular_solution[col] = rref_matrix[i][n_vars]
            
            # Find homogeneous solutions (one for each free variable)
            homogeneous_solutions = []
            for free_var in free_vars:
                homo_solution = [Fraction(0)] * n_vars
                homo_solution[free_var] = Fraction(1)  # Set this free variable to 1
                
                # Express pivot variables in terms of this free variable
                for i, col in enumerate(pivot_cols):
                    if i < len(rref_matrix) and free_var < len(rref_matrix[i]):
                        homo_solution[col] = -rref_matrix[i][free_var]
                
                homogeneous_solutions.append(homo_solution)
            
            return {
                'type': 'infinite',
                'particular_solution': particular_solution,
                'homogeneous_solutions': homogeneous_solutions,
                'free_vars': free_vars,
                'pivot_cols': pivot_cols
            }
    
    except Exception as e:
        print(f"Error finding general solution: {e}")
        return None

def display_general_solution(solution_info, n_vars):
    """Display the general solution in a clear mathematical format."""
    if solution_info is None:
        print("Error: Could not determine solution")
        return
    
    if solution_info.get('inconsistent', False):
        print("System is INCONSISTENT - No solution exists")
        print(f"Contradiction: {solution_info['impossible_equation']}")
        return
    
    if solution_info['type'] == 'unique':
        print("System has a UNIQUE solution:")
        print("\nSolution vector x:")
        for i, val in enumerate(solution_info['solution']):
            print(f"  x{i+1} = {format_fraction(val)}")
    
    elif solution_info['type'] == 'infinite':
        print("System has INFINITE solutions:")
        print(f"Free variables: {[f'x{i+1}' for i in solution_info['free_vars']]}")
        print(f"Pivot variables: {[f'x{i+1}' for i in solution_info['pivot_cols']]}")
        
        # Display general solution
        print("\nGeneral solution:")
        
        # Build particular solution vector
        particular_str = "[" + ", ".join(format_fraction(val) for val in solution_info['particular_solution']) + "]"
        
        # Build homogeneous solution vectors
        free_var_names = [f"t{i+1}" for i in range(len(solution_info['free_vars']))]
        homo_strs = []
        
        for i, homo_sol in enumerate(solution_info['homogeneous_solutions']):
            homo_str = f"{free_var_names[i]}[" + ", ".join(format_fraction(val) for val in homo_sol) + "]"
            homo_strs.append(homo_str)
        
        # Print the complete general solution
        if all(val == 0 for val in solution_info['particular_solution']):
            print("x = " + " + ".join(homo_strs))
        else:
            print("x = " + particular_str + " + " + " + ".join(homo_strs))
        
        print(f"\nwhere {', '.join(free_var_names)} can be any real numbers.")
        
        # Show component form
        print("\nComponent form:")
        for i in range(n_vars):
            expr_parts = []
            
            # Add particular solution part
            if solution_info['particular_solution'][i] != 0:
                expr_parts.append(format_fraction(solution_info['particular_solution'][i]))
            
            # Add homogeneous parts
            for j, (homo_sol, param) in enumerate(zip(solution_info['homogeneous_solutions'], free_var_names)):
                coeff = homo_sol[i]
                if coeff != 0:
                    if coeff == 1:
                        expr_parts.append(param)
                    elif coeff == -1:
                        expr_parts.append(f"-{param}")
                    else:
                        expr_parts.append(f"{format_fraction(coeff)}{param}")
            
            if not expr_parts:
                expr = "0"
            else:
                expr = expr_parts[0]
                for part in expr_parts[1:]:
                    if part.startswith('-'):
                        expr += f" {part}"
                    else:
                        expr += f" + {part}"
            
            print(f"  x{i+1} = {expr}")

def safe_pivot_search(augmented, current_row, col, m):
    """Safely find the best pivot row with bounds checking."""
    if current_row >= m or col >= augmented.shape[1]:
        return current_row
    
    pivot_row = current_row
    max_val = abs(augmented[current_row, col])
    
    for row in range(current_row + 1, m):
        val = abs(augmented[row, col])
        if val > max_val:
            max_val = val
            pivot_row = row
    
    return pivot_row

def print_operation_header(operation, step_num):
    """Print a concise header for each row operation."""
    print(f"\nStep {step_num}: {operation}")

def reduced_row_echelon_form(A, b, show_steps=False):
    """Transform the augmented matrix [A|b] to RREF with detailed step-by-step operations."""
    try:
        A, b = validate_input(A, b)
        augmented = np.column_stack([A, b])
        m, n = augmented.shape
        n_vars = n - 1
        
        pivot_cols = []
        current_row = 0
        step_count = 0
        
        if show_steps:
            step_count += 1
            print_operation_header("Initial matrix [A|b]", step_count)
            print_matrix(augmented)
            if m > 5 or n_vars > 5:
                print(f"Warning: Large matrix ({m}×{n_vars}) - output will be verbose")
        
        # Forward elimination to row echelon form
        for col in range(n_vars):
            if current_row >= m:
                break
                
            # Find pivot with safety checks
            pivot_row = safe_pivot_search(augmented, current_row, col, m)
            
            # Skip if column is effectively zero
            if abs(augmented[pivot_row, col]) < Config.ZERO_THRESHOLD:
                if show_steps:
                    print(f"\nSkipping column {col+1} - all entries are effectively zero")
                continue
            
            # Swap rows if needed
            if pivot_row != current_row:
                step_count += 1
                if show_steps:
                    print_operation_header(f"Swap R{current_row+1} ↔ R{pivot_row+1}", step_count)
                
                augmented[[current_row, pivot_row]] = augmented[[pivot_row, current_row]]
                
                if show_steps:
                    print_matrix(augmented)
            
            pivot_cols.append(col)
            pivot_val = augmented[current_row, col]
            
            # Protect against division by very small numbers
            if abs(pivot_val) < Config.ZERO_THRESHOLD:
                continue
            
            # Scale pivot row to make pivot = 1
            if abs(pivot_val - 1.0) > Config.ZERO_THRESHOLD:
                step_count += 1
                if show_steps:
                    print_operation_header(f"Scale R{current_row+1} by {format_number(1/pivot_val)}", step_count)
                
                augmented[current_row] = augmented[current_row] / pivot_val
                
                if show_steps:
                    print_matrix(augmented)
            
            # Eliminate column
            elimination_performed = False
            for row in range(m):
                if row != current_row and abs(augmented[row, col]) > Config.ZERO_THRESHOLD:
                    if not elimination_performed and show_steps:
                        step_count += 1
                        print_operation_header(f"Eliminate column {col+1}", step_count)
                        elimination_performed = True
                    
                    multiplier = augmented[row, col]
                    if show_steps:
                        print(f"R{row+1} = R{row+1} - ({format_number(multiplier)}) × R{current_row+1}")
                    
                    augmented[row] = augmented[row] - multiplier * augmented[current_row]
            
            if elimination_performed and show_steps:
                print_matrix(augmented)
            
            current_row += 1
        
        if show_steps:
            step_count += 1
            print_operation_header("Final RREF", step_count)
            print_matrix(augmented)
            print(f"Steps: {step_count}, Rank: {len(pivot_cols)}, Pivots: {[f'x{i}' for i in pivot_cols]}")
            if len(pivot_cols) < n_vars:
                print(f"Free variables: {n_vars - len(pivot_cols)}")
        
        return augmented, pivot_cols
        
    except Exception as e:
        print(f"Error in RREF computation: {e}")
        return None, []

def analyze_rref(rref_matrix, pivot_cols, n_vars):
    """Analyze the RREF matrix to determine system properties with error handling."""
    if rref_matrix is None:
        return {'rank': 0, 'inconsistent': True, 'system_type': 'ERROR', 
                'description': 'Error in RREF computation', 'pivot_cols': [], 'free_vars': 0}
    
    try:
        m, n = rref_matrix.shape
        rank = len(pivot_cols)
        
        # Check for inconsistency
        inconsistent = False
        for i in range(m):
            var_part = rref_matrix[i, :n_vars]
            b_part = rref_matrix[i, n_vars] if n_vars < n else 0
            if np.allclose(var_part, 0, atol=Config.ZERO_THRESHOLD) and abs(b_part) > Config.ZERO_THRESHOLD:
                inconsistent = True
                break
        
        # Determine system type
        if inconsistent:
            system_type = "INCONSISTENT"
            description = "No solution exists"
        elif rank == n_vars:
            system_type = "UNIQUE"
            description = "Unique solution"
        else:
            system_type = "INFINITE"
            free_vars = n_vars - rank
            description = f"Infinite solutions ({free_vars} free variables)"
        
        return {
            'rank': rank,
            'inconsistent': inconsistent,
            'system_type': system_type,
            'description': description,
            'pivot_cols': pivot_cols,
            'free_vars': n_vars - rank if not inconsistent else 0
        }
        
    except Exception as e:
        print(f"Error analyzing RREF: {e}")
        return {'rank': 0, 'inconsistent': True, 'system_type': 'ERROR', 
                'description': f'Analysis error: {e}', 'pivot_cols': [], 'free_vars': 0}

def get_system_examples():
    """Return example systems for testing - separated for maintainability."""
    examples = {
        'inconsistent': {
            'A': np.array([
                [1, -1, -3, -1],
                [-2, 2, 6, 2],
                [-3, -3, 10, 0]
            ]),
            'b': np.array([-1, -1, 5]),
            'description': 'Inconsistent system (no solution)'
        },
        'square': {
            'A': np.array([
                [2, -1, 0, 0],
                [-1, 2, -1, 0],
                [0, -1, 2, -1],
                [0, 0, -1, 2]
            ]),
            'b': np.array([1, 0, 0, 1]),
            'description': 'Well-conditioned square system'
        },
        'underdetermined': {
            'A': np.array([
                [1, 2, -1, 3],
                [2, 4, 1, 1]
            ]),
            'b': np.array([1, 4]),
            'description': 'Underdetermined system (infinite solutions)'
        }
    }
    return examples

def perform_exact_analysis(A, b, show_rref=True, show_steps=False):
    """Perform exact RREF analysis and find the complete general solution."""
    if not show_rref:
        return None
    
    print("\nExact RREF Analysis:")
    
    try:
        # Convert input to exact fractions and perform RREF
        rref_matrix, pivot_cols = exact_rref(A, b, show_steps=show_steps)
        
        if rref_matrix is not None:
            if not show_steps:
                print("RREF (exact fractions):")
                print_matrix_exact(rref_matrix)
                print()
            
            n_vars = len(A[0])
            
            # Find and display general solution
            solution_info = find_general_solution(rref_matrix, pivot_cols, n_vars)
            
            print(f"Rank: {len(pivot_cols)}")
            print(f"Variables: {n_vars}")
            print(f"Pivots: {[f'x{col+1}' for col in pivot_cols]}")
            
            if solution_info:
                try:
                    display_general_solution(solution_info, n_vars)
                except Exception as e:
                    print(f"Error displaying solution: {e}")
                    print(f"Solution info: {solution_info}")
            
            return solution_info
        else:
            print("Error: Could not compute exact RREF")
            return None
            
    except Exception as e:
        print(f"Error in exact analysis: {e}")
        return None

def perform_rref_analysis(A, b, show_rref=True, show_steps=False):
    """Perform and display RREF analysis with optional step-by-step operations."""
    if not show_rref:
        return None
    
    print("\nRREF Analysis:")
    
    try:
        if not show_steps:
            print("Original [A|b]:")
            augmented_orig = np.column_stack([A, b])
            print_matrix(augmented_orig)
            print()
        
        rref_matrix, pivot_cols = reduced_row_echelon_form(A, b, show_steps=show_steps)
        
        if rref_matrix is not None:
            if not show_steps:
                print("RREF:")
                print_matrix(rref_matrix)
                print()
            
            analysis = analyze_rref(rref_matrix, pivot_cols, A.shape[1])
            
            print(f"Rank: {analysis['rank']}")
            print(f"Pivots: {[f'x{col}' for col in pivot_cols]}")
            print(f"Type: {analysis['description']}")
            
            if analysis['system_type'] == "INFINITE":
                free_vars = [i for i in range(A.shape[1]) if i not in pivot_cols]
                print(f"Free variables: {[f'x{i}' for i in free_vars]}")
                
            elif analysis['system_type'] == "INCONSISTENT":
                print("Inconsistent: No solution exists")
                m, n_aug = rref_matrix.shape
                for i in range(m):
                    var_part = rref_matrix[i, :A.shape[1]]
                    b_part = rref_matrix[i, A.shape[1]]
                    if np.allclose(var_part, 0, atol=Config.ZERO_THRESHOLD) and abs(b_part) > Config.ZERO_THRESHOLD:
                        print(f"Impossible: 0 = {format_number(b_part)}")
                        break
                        
            elif analysis['system_type'] == "UNIQUE":
                print("Unique solution exists")
            
            return analysis
        else:
            print("Error: Could not compute RREF")
            return None
            
    except Exception as e:
        print(f"Error in RREF analysis: {e}")
        return None

def check_system_consistency(A, b):
    """Check system consistency using rank analysis."""
    try:
        rank_A = np.linalg.matrix_rank(A)
        rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
        return rank_A, rank_Ab, rank_A == rank_Ab
    except Exception as e:
        print(f"Error checking consistency: {e}")
        return 0, 0, False

def handle_inconsistent_system(A, b, rank_A, rank_Ab, auto_proceed=False):
    """Handle inconsistent system detection and user interaction."""
    print("Inconsistent system detected:")
    print(f"rank(A) = {rank_A}, rank([A|b]) = {rank_Ab}")
    print("No exact solution exists.")
    
    if auto_proceed:
        print("Auto-proceeding with least-squares approximation...")
        proceed = True
    else:
        try:
            user_choice = input("Proceed with least-squares approximation? (y/n): ").lower().strip()
            proceed = user_choice in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            print("Operation cancelled.")
            return None, None
    
    if proceed:
        print("Computing least-squares approximation...")
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return x, "Least Squares Approximation"
        except np.linalg.LinAlgError as e:
            print(f"Error: {e}")
            return None, None
    else:
        print("Stopping calculation.")
        return None, None

def solve_consistent_system(A, b, system_type, condition_number, mpmath_precision):
    """Solve consistent system using appropriate method."""
    try:
        if system_type == "UNDERDETERMINED":
            print("Finding minimum norm solution...")
            A_pinv = np.linalg.pinv(A)
            x = A_pinv @ b
            return x, "Minimum Norm Solution"
        
        else:
            # Square or overdetermined system
            print("Applying matrix equilibration...")
            
            def equilibrate_matrix(A, b):
                """Apply row and column scaling for better conditioning."""
                row_norms = np.sqrt(np.sum(A**2, axis=1))
                col_norms = np.sqrt(np.sum(A**2, axis=0))
                
                row_scales = np.where(row_norms > Config.ZERO_THRESHOLD, 1.0 / row_norms, 1.0)
                col_scales = np.where(col_norms > Config.ZERO_THRESHOLD, 1.0 / col_norms, 1.0)
                
                A_equilibrated = np.diag(row_scales) @ A @ np.diag(col_scales)
                b_equilibrated = row_scales * b
                
                return A_equilibrated, b_equilibrated, row_scales, col_scales
            
            A_eq, b_eq, row_scales, col_scales = equilibrate_matrix(A, b)
            equilibrated_cond = np.linalg.cond(A_eq)
            
            if np.isfinite(condition_number) and np.isfinite(equilibrated_cond):
                improvement = condition_number / equilibrated_cond
                print(f"Condition number improved by {improvement:.1f}×")
            
            # Check precision options
            mpmath_available = False
            try:
                import mpmath as mp
                mpmath_available = True
            except ImportError:
                pass
            
            float128_available = False
            try:
                test_hp = np.array([1.0], dtype=np.float128)
                float128_available = True
            except:
                pass
            
            # Solve with best available precision
            if mpmath_available and condition_number > Config.HIGH_CONDITION_THRESHOLD and system_type == "SQUARE":
                print("Using arbitrary precision (mpmath)...")
                try:
                    mp.mp.dps = mpmath_precision
                    A_mp = mp.matrix(A_eq.tolist())
                    b_mp = mp.matrix(b_eq.tolist())
                    y_mp = mp.lu_solve(A_mp, b_mp)
                    y_result = np.array([float(y_mp[i]) for i in range(len(b_eq))])
                    x = col_scales * y_result
                    return x, "Equilibration + Arbitrary Precision"
                except Exception as e:
                    print(f"Mpmath failed ({e}), falling back to standard precision...")
            
            elif float128_available and condition_number > Config.HIGH_CONDITION_THRESHOLD:
                print("Using extended precision (float128)...")
                try:
                    if system_type == "SQUARE":
                        A_hp = A_eq.astype(np.float128)
                        b_hp = b_eq.astype(np.float128)
                        y_hp = np.linalg.solve(A_hp, b_hp)
                        x = (col_scales * y_hp).astype(np.float64)
                    else:
                        x, _, _, _ = np.linalg.lstsq(A.astype(np.float128), b.astype(np.float128), rcond=None)
                        x = x.astype(np.float64)
                    return x, "Equilibration + Extended Precision"
                except Exception as e:
                    print(f"Float128 failed ({e}), falling back to standard precision...")
            
            # Standard precision solution
            try:
                if system_type == "SQUARE":
                    y = np.linalg.solve(A_eq, b_eq)
                    x = col_scales * y
                    return x, "Equilibration + Standard Precision"
                else:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    return x, "Least Squares Solution"
            except np.linalg.LinAlgError as e:
                print(f"Standard solution failed: {e}")
                return None, None
                
    except Exception as e:
        print(f"Error solving system: {e}")
        return None, None

def evaluate_solution_quality(residual_norm):
    """Evaluate solution quality based on residual norm."""
    if residual_norm < Config.EXCELLENT_QUALITY:
        return "EXCELLENT"
    elif residual_norm < Config.GOOD_QUALITY:
        return "GOOD"
    elif residual_norm < Config.ACCEPTABLE_QUALITY:
        return "ACCEPTABLE"
    else:
        return "POOR"

def display_solution_results(x, A, b, method, system_type, rank_A, mpmath_available):
    """Display solution results with verification."""
    try:
        # Solution verification
        residual = A @ x - b
        residual_norm = np.linalg.norm(residual)
        quality = evaluate_solution_quality(residual_norm)
        
        print(f"Residual norm: {residual_norm:.2e}")
        print(f"Quality: {quality}")
        
        print("\nSolution:")
        
        # Display the solution
        print("Solution vector x:")
        for i, xi in enumerate(x):
            print(f"  x[{i}] = {format_number(xi)}")
        
        print(f"\nMethod: {method}")
        print(f"Solution norm: {format_number(np.linalg.norm(x))}")
        
        # Concise verification
        print("\nVerification (Ax vs b):")
        for i in range(len(b)):
            computed = np.dot(A[i], x)
            error = abs(computed - b[i])
            print(f"  Eq {i+1}: {format_number(computed)} vs {format_number(b[i])} (error: {error:.1e})")
        
        # Brief recommendations
        if system_type == "UNDERDETERMINED":
            print(f"\nNote: Minimum norm solution shown. {A.shape[1] - rank_A} free parameters exist.")
        elif residual_norm > Config.GOOD_QUALITY:
            print("\nFor better accuracy, consider:")
            if mpmath_available:
                print("- Increase precision: solve_your_system(mpmath_precision=50)")
            else:
                print("- Install mpmath: pip install mpmath")
        
        return {
            'solution': x,
            'residual_norm': residual_norm,
            'quality': quality,
            'method': method,
            'system_type': system_type,
            'rank': rank_A
        }
        
    except Exception as e:
        print(f"Error displaying results: {e}")
        return None

def solve_your_system(A_input, b_input, mpmath_precision=Config.DEFAULT_MPMATH_PRECISION, show_rref=True, show_steps=False, auto_proceed_inconsistent=False):
    """
    Main solver function - finds ALL solutions using exact arithmetic.
    
    Parameters:
    - A_input: Your coefficient matrix from the top of the file
    - b_input: Your right-hand side vector from the top of the file  
    - mpmath_precision: Decimal digits for arbitrary precision
    - show_rref: Whether to show RREF analysis
    - show_steps: Whether to show detailed row operations
    - auto_proceed_inconsistent: Auto-proceed with least squares for inconsistent systems
    """
    
    # Convert user input to numpy arrays
    A = np.array(A_input, dtype=float)
    b = np.array(b_input, dtype=float)
    print("Finding ALL solutions using exact arithmetic")
    
    try:
        # Input validation
        A, b = validate_input(A, b)
        
        print(f"Matrix shape: {A.shape}")
        
        # Determine system type
        m, n = A.shape
        if m == n:
            system_type = "SQUARE"
            print("System type: Square system")
        elif m > n:
            system_type = "OVERDETERMINED"
            print("System type: Overdetermined system")
        else:
            system_type = "UNDERDETERMINED"
            print("System type: Underdetermined system")
        
        # EXACT Analysis - this is the main improvement
        exact_solution = perform_exact_analysis(A_input, b_input, show_rref, show_steps)
        
        return exact_solution
        
    except Exception as e:
        print(f"Critical error in solve_your_system: {e}")
        return None

if __name__ == "__main__":
    try:
        print("EXACT LINEAR SYSTEM SOLVER")
        print("=" * 50)
        
        result = solve_your_system(
            A_input=A,  # Uses your matrix A from the top of the file
            b_input=b,  # Uses your vector b from the top of the file
            mpmath_precision=precision, 
            show_rref=show_rref,
            show_steps=show_steps,
            auto_proceed_inconsistent=auto_proceed
        )
        
        if result is not None:
            print(f"\nAnalysis complete: {result.get('type', 'unknown')} solution type")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 