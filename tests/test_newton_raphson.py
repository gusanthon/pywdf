import math
import numpy as np

import sys
from pathlib import Path

# Allow direct execution: python tests/test_newton_raphson.py
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pywdf.core.solver.newton_raphson import newton_raphson
# Example 1: Find the square root of 2 (root of x^2 - 2 = 0)
def f1(x):
    return x**2 - 2

def df1(x):
    return 2*x

print("Example 1: Finding square root of 2")
root1, iterations1 = newton_raphson(f1, df1, x0=1.0)
print(f"Root: {root1:.10f}")
print(f"Iterations: {iterations1}")
print(f"Verification: {root1}^2 = {root1**2:.10f}")
print()

# Example 2: Find root of x^3 - x - 1 = 0
def f2(x):
    return x**3 - x - 1

def df2(x):
    return 3*x**2 - 1

print("Example 2: Finding root of x^3 - x - 1 = 0")
root2, iterations2 = newton_raphson(f2, df2, x0=1.5)
print(f"Root: {root2:.10f}")
print(f"Iterations: {iterations2}")
print(f"Verification: f({root2:.6f}) = {f2(root2):.2e}")
print()

# Example 3: Find root of cos(x) - x = 0
def f3(x):
    return math.cos(x) - x

def df3(x):
    return -math.sin(x) - 1

print("Example 3: Finding root of cos(x) - x = 0")
root3, iterations3 = newton_raphson(f3, df3, x0=0.5)
print(f"Root: {root3:.10f}")
print(f"Iterations: {iterations3}")
print(f"Verification: cos({root3:.6f}) - {root3:.6f} = {f3(root3):.2e}")
print()

# Example with plotting (optional visualization)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_newton_raphson_steps(f, df, x0, num_steps=5):
        """Visualize the Newton-Raphson iteration steps"""
        x_vals = [x0]
        x = x0
        
        # Perform iterations and collect x values
        for _ in range(num_steps):
            x = x - f(x) / df(x)
            x_vals.append(x)
        
        # Create plot
        x_plot = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 1000)
        y_plot = [f(xi) for xi in x_plot]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, 'b-', label='f(x) = x² - 2')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot iteration steps
        colors = ['red', 'orange', 'green', 'purple', 'brown']
        for i in range(min(len(x_vals)-1, len(colors))):
            x_curr = x_vals[i]
            # Tangent line
            slope = df(x_curr)
            y_tangent = slope * (x_plot - x_curr) + f(x_curr)
            plt.plot(x_plot, y_tangent, '--', color=colors[i], alpha=0.7, 
                    label=f'Iteration {i+1}')
            # Mark the point
            plt.plot(x_curr, f(x_curr), 'o', color=colors[i], markersize=8)
            plt.plot(x_vals[i+1], 0, 's', color=colors[i], markersize=6)
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Newton-Raphson Method Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot to file
        plt.savefig('newton_raphson_visualization.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to 'newton_raphson_visualization.png'")
        plt.show()
    
    print("Plotting Newton-Raphson steps for x² - 2 = 0...")
    plot_newton_raphson_steps(f1, df1, 1.0)
    print("Plotted Newton-Raphson steps for x² - 2 = 0...")
    
except ImportError:
    print("Matplotlib not available for plotting visualization")