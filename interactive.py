#!/usr/bin/env python3
"""
Interactive command-line interface for the Gaussian elimination solver.

This script provides a user-friendly way to solve linear systems with 
step-by-step explanations and multiple input options.
"""

import numpy as np
import sys
from typing import Optional, Tuple
from gaussian_elimination import solve, print_system, print_matrix, format_solution
from gaussian_elimination.utils import print_augmented_matrix
from gaussian_elimination.core import get_solution_info


def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header."""
    print("=" * 70)
    print("🔢 INTERACTIVE GAUSSIAN ELIMINATION SOLVER")
    print("=" * 70)
    print("Solve linear systems Ax = b with step-by-step explanations")
    print("=" * 70)
    print()


def print_menu():
    """Print the main menu options."""
    print("📋 MAIN MENU")
    print("-" * 30)
    print("1. Enter your own matrix and vector")
    print("2. Choose from example systems")
    print("3. Learn about Gaussian elimination")
    print("4. Run comprehensive demonstrations")
    print("5. Exit")
    print("-" * 30)


def get_user_choice() -> str:
    """Get user's menu choice."""
    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def input_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """Get matrix A and vector b from user input."""
    print("\n📝 ENTER YOUR LINEAR SYSTEM")
    print("-" * 40)
    print("You can now enter rectangular matrices (any size)!")
    print("• Square (n×n): May have unique solution")
    print("• Overdetermined (n>m): More equations than unknowns")  
    print("• Underdetermined (n<m): More unknowns than equations")
    print()
    
    # Get number of equations (rows)
    while True:
        try:
            n = int(input("Enter the number of equations (rows): "))
            if n <= 0:
                print("❌ Number of equations must be positive.")
                continue
            if n > 10:
                print("❌ Maximum number of equations is 10 for this interface.")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid integer.")
    
    # Get number of variables (columns)
    while True:
        try:
            m = int(input("Enter the number of variables (columns): "))
            if m <= 0:
                print("❌ Number of variables must be positive.")
                continue
            if m > 10:
                print("❌ Maximum number of variables is 10 for this interface.")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid integer.")
    
    # Show system type
    if n == m:
        system_type = "Square system (unique solution expected)"
    elif n > m:
        system_type = "Overdetermined system (may have no solution or unique solution)"
    else:
        system_type = "Underdetermined system (infinite solutions expected)"
    
    print(f"\n🔍 System type: {system_type}")
    
    print(f"\n📊 Enter the coefficient matrix A ({n}×{m}):")
    print("Enter each row on a separate line, with numbers separated by spaces.")
    if n == 2 and m == 2:
        print("Example for 2×2: '2 3' then '1 -1'")
    elif n == 3 and m == 2:
        print("Example for 3×2: '1 2' then '3 4' then '5 6'")
    elif n == 2 and m == 3:
        print("Example for 2×3: '1 2 3' then '4 5 6'")
    print()
    
    A = np.zeros((n, m), dtype=float)
    for i in range(n):
        while True:
            try:
                row_input = input(f"Row {i+1}: ").strip()
                row_values = [float(x) for x in row_input.split()]
                if len(row_values) != m:
                    print(f"❌ Please enter exactly {m} numbers.")
                    continue
                A[i] = row_values
                break
            except ValueError:
                print("❌ Please enter valid numbers separated by spaces.")
    
    print(f"\n📊 Enter the right-hand side vector b ({n} elements):")
    print("Enter all numbers on one line, separated by spaces.")
    if n == 2:
        print("Example: '7 1'")
    elif n == 3:
        print("Example: '7 1 5'")
    
    while True:
        try:
            b_input = input("Vector b: ").strip()
            b_values = [float(x) for x in b_input.split()]
            if len(b_values) != n:
                print(f"❌ Please enter exactly {n} numbers.")
                continue
            b = np.array(b_values, dtype=float)
            break
        except ValueError:
            print("❌ Please enter valid numbers separated by spaces.")
    
    return A, b


def show_example_menu() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Show menu of example systems and return the selected one."""
    examples = {
        "1": {
            "name": "Simple 2×2 System (Unique Solution)",
            "description": "Square: 2x + 3y = 7, x - y = 1",
            "A": np.array([[2, 3], [1, -1]], dtype=float),
            "b": np.array([7, 1], dtype=float)
        },
        "2": {
            "name": "3×3 System (Unique Solution)",
            "description": "Square: System requiring partial pivoting",
            "A": np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float),
            "b": np.array([8, -11, -3], dtype=float)
        },
        "3": {
            "name": "Overdetermined 3×2 System (Consistent)",
            "description": "More equations than unknowns - has unique solution",
            "A": np.array([[1, 1], [1, 2], [2, 3]], dtype=float),
            "b": np.array([3, 4, 7], dtype=float)
        },
        "4": {
            "name": "Overdetermined 3×2 System (Inconsistent)",
            "description": "More equations than unknowns - no solution",
            "A": np.array([[1, 1], [1, 1], [1, 1]], dtype=float),
            "b": np.array([2, 3, 4], dtype=float)
        },
        "5": {
            "name": "Underdetermined 2×3 System",
            "description": "More unknowns than equations - infinite solutions",
            "A": np.array([[1, 2, 3], [2, 4, 6]], dtype=float),
            "b": np.array([6, 12], dtype=float)
        },
        "6": {
            "name": "System Requiring Pivoting",
            "description": "Square: First element is zero, needs row swapping",
            "A": np.array([[0, 2, 3], [1, 1, 1], [2, 1, 1]], dtype=float),
            "b": np.array([13, 6, 5], dtype=float)
        },
        "7": {
            "name": "No Solution System (Square)",
            "description": "Square: Inconsistent system (contradictory equations)",
            "A": np.array([[1, 1], [1, 1]], dtype=float),
            "b": np.array([1, 2], dtype=float)
        },
        "8": {
            "name": "Infinite Solutions System (Square)",
            "description": "Square: Dependent equations (rank deficient)",
            "A": np.array([[1, 2], [2, 4]], dtype=float),
            "b": np.array([3, 6], dtype=float)
        }
    }
    
    print("\n📚 EXAMPLE SYSTEMS")
    print("-" * 50)
    print("SQUARE SYSTEMS:")
    for key in ["1", "2", "6", "7", "8"]:
        example = examples[key]
        print(f"{key}. {example['name']}")
        print(f"   {example['description']}")
        print()
    
    print("RECTANGULAR SYSTEMS:")
    for key in ["3", "4", "5"]:
        example = examples[key]
        print(f"{key}. {example['name']}")
        print(f"   {example['description']}")
        print()
    
    print("9. Return to main menu")
    print("-" * 50)
    
    while True:
        choice = input("Choose an example (1-9): ").strip()
        if choice in examples:
            return examples[choice]["A"], examples[choice]["b"]
        elif choice == "9":
            return None
        else:
            print("❌ Invalid choice. Please enter 1-9.")


def solve_and_display(A: np.ndarray, b: np.ndarray):
    """Solve the system and display results with explanations."""
    print("\n" + "=" * 60)
    print("🔍 ANALYZING YOUR SYSTEM")
    print("=" * 60)
    
    n, m = A.shape
    
    # Show system information
    print(f"System dimensions: {n} equations, {m} unknowns")
    if n == m:
        system_type = "Square system"
        expected = "May have unique solution, no solution, or infinite solutions"
    elif n > m:
        system_type = "Overdetermined system"
        expected = "More equations than unknowns - may have no solution or unique solution"
    else:
        system_type = "Underdetermined system"
        expected = "More unknowns than equations - likely infinite solutions"
    
    print(f"Type: {system_type}")
    print(f"Expected: {expected}")
    print()
    
    # Display the system
    print("Your system of equations:")
    try:
        print_system(A, b)
    except Exception:
        print("System display error - continuing with matrix form...")
    
    print(f"Coefficient matrix A ({n}×{m}):")
    print_matrix(A, precision=3)
    
    print("Right-hand side vector b:")
    print_matrix(b.reshape(1, -1), precision=3)
    
    print("Augmented matrix [A|b]:")
    print_augmented_matrix(A, b)
    
    # Solve the system
    print("🧮 SOLVING USING GAUSSIAN ELIMINATION")
    print("-" * 60)
    
    try:
        solution_type, solution = get_solution_info(A, b)
        
        if solution_type == "unique":
            print("✅ SOLUTION FOUND!")
            if n == m:
                print("The square system has a unique solution.")
            elif n > m:
                print("The overdetermined system is consistent and has a unique solution.")
                print("This means all equations are compatible with each other.")
            else:
                print("Note: This is unexpected for an underdetermined system.")
            print()
            print(f"Solution: {format_solution(solution)}")
            
            # Verification
            print("\n🔍 VERIFICATION")
            print("-" * 30)
            result = np.dot(A, solution)
            print("Computing A × x:")
            print_matrix(result.reshape(1, -1), "A × x", precision=6)
            print_matrix(b.reshape(1, -1), "b", precision=6)
            
            difference = np.abs(result - b)
            max_error = np.max(difference)
            print(f"Maximum absolute error: {max_error:.2e}")
            
            if np.allclose(result, b, rtol=1e-10):
                print("✅ Verification successful! The solution is correct.")
                if n > m:
                    print("🎯 All overdetermined equations are satisfied exactly!")
            else:
                print("⚠️  Large numerical errors detected.")
                
        elif solution_type == "no_solution":
            print("❌ NO SOLUTION")
            print("The system is inconsistent (contradictory equations).")
            if n > m:
                print("This is common for overdetermined systems where you have")
                print("more equations than unknowns and they contradict each other.")
            elif n == m:
                print("This happens when equations in the square system contradict.")
            else:
                print("This is rare for underdetermined systems.")
            print("\nTechnical: rank([A|b]) > rank(A)")
            
        elif solution_type == "infinite_solutions":
            print("♾️  INFINITE SOLUTIONS")
            if n < m:
                print("This is expected for underdetermined systems.")
                print("You have more unknowns than equations, so there are")
                print("free variables that can take any value.")
            elif n == m:
                print("The square system has dependent equations (rank deficient).")
                print("Some equations are linear combinations of others.")
            else:
                print("This is rare for overdetermined systems but can happen")
                print("when extra equations are dependent on the first m equations.")
            print("\nTechnical: rank(A) < number of variables")
            
        else:
            print("❓ UNEXPECTED ERROR")
            print("An unexpected error occurred during solving.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 60)


def show_educational_content():
    """Display educational information about Gaussian elimination."""
    print("\n📖 ABOUT GAUSSIAN ELIMINATION")
    print("=" * 60)
    
    content = """
🎯 WHAT IS GAUSSIAN ELIMINATION?
Gaussian elimination is a systematic method for solving systems of linear 
equations by transforming the coefficient matrix into row echelon form.
Now supports rectangular matrices of any size!

📐 TYPES OF LINEAR SYSTEMS:

1. SQUARE SYSTEMS (n = m equations and unknowns):
   ✅ Unique solution: Full rank matrix
   ❌ No solution: Inconsistent equations  
   ♾️  Infinite solutions: Rank deficient matrix

2. OVERDETERMINED SYSTEMS (n > m, more equations than unknowns):
   ✅ Unique solution: Consistent and full column rank
   ❌ No solution: Inconsistent (most common case)
   ♾️  Infinite solutions: Rare, requires special structure

3. UNDERDETERMINED SYSTEMS (n < m, more unknowns than equations):
   ♾️  Infinite solutions: Always (if consistent)
   ❌ No solution: Only if inconsistent

🔄 THE ALGORITHM STEPS:
1. FORWARD ELIMINATION: Transform matrix to row echelon form
   - Perform min(n,m) elimination steps
   - Use row operations to eliminate elements below pivots
   - Apply partial pivoting for numerical stability

2. SOLUTION ANALYSIS: Check rank conditions
   - rank(A) vs rank([A|b]) for consistency
   - rank(A) vs number of variables for uniqueness

3. BACK SUBSTITUTION: Solve for variables (when unique solution exists)
   - Start with pivot equations and work backwards
   - Handle rectangular cases appropriately

🔄 PARTIAL PIVOTING:
- Find the row with largest absolute value in current column
- Swap rows to bring this element to the diagonal position
- Prevents division by small numbers and improves stability

📊 RANK CONDITIONS:
- rank([A|b]) > rank(A) → No solution (inconsistent)
- rank(A) < number of variables → Infinite solutions
- rank(A) = number of variables = number of equations → Unique solution

🧮 COMPLEXITY:
Time complexity: O(n²m) for n×m matrix (O(n³) when square)
Space complexity: O(nm) for storing the matrix

📚 APPLICATIONS:
- Engineering: Circuit analysis, structural mechanics, fluid dynamics
- Data Science: Linear regression, least squares fitting  
- Computer Graphics: 3D transformations, rendering
- Economics: Input-output models, linear programming
- Machine Learning: Neural networks, optimization
"""
    
    print(content)
    
    input("\nPress Enter to continue...")


def run_demonstrations():
    """Run the comprehensive demonstrations from main.py."""
    print("\n🎬 RUNNING COMPREHENSIVE DEMONSTRATIONS")
    print("=" * 60)
    print("This will show various examples with detailed explanations...")
    print()
    
    confirm = input("Do you want to run the demonstrations? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        try:
            from main import main as run_main_demo
            print("\n" + "=" * 60)
            run_main_demo()
            print("\n" + "=" * 60)
            input("Press Enter to return to the main menu...")
        except ImportError:
            print("❌ Could not import demonstration module.")
    else:
        print("Returning to main menu...")


def main():
    """Main interactive loop."""
    try:
        while True:
            clear_screen()
            print_header()
            print_menu()
            
            choice = get_user_choice()
            
            if choice == "1":
                # User input
                try:
                    A, b = input_matrix()
                    solve_and_display(A, b)
                    input("\nPress Enter to continue...")
                except KeyboardInterrupt:
                    print("\n\nReturning to main menu...")
                    continue
                
            elif choice == "2":
                # Example systems
                result = show_example_menu()
                if result is not None:
                    A, b = result
                    solve_and_display(A, b)
                    input("\nPress Enter to continue...")
                
            elif choice == "3":
                # Educational content
                show_educational_content()
                
            elif choice == "4":
                # Demonstrations
                run_demonstrations()
                
            elif choice == "5":
                # Exit
                print("\n👋 Thank you for using the Gaussian Elimination Solver!")
                print("Happy computing! 🧮")
                break
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Please restart the application.")
        sys.exit(1)


if __name__ == "__main__":
    main() 