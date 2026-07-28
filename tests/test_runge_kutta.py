import math
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Allow direct execution: python tests/test_runge_kutta.py
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pywdf.core.solver.runge_kutta import rk4_solve

# Example 1: Simple exponential decay dy/dt = -2y, y(0) = 1
def example1():
    def f(t, y):
        return -2 * y
    
    # Analytical solution: y = e^(-2t)
    def analytical(t):
        return np.exp(-2 * t)
    
    t_span = (0, 2)
    y0 = 1
    h = 0.1
    
    t_num, y_num = rk4_solve(f, t_span, y0, h)
    
    t_exact = np.linspace(0, 2, 100)
    y_exact = analytical(t_exact)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_exact, y_exact, 'b-', label='Analytical solution', linewidth=2)
    plt.plot(t_num, y_num, 'ro-', label='RK4 numerical solution', markersize=4)
    plt.xlabel('Time t')
    plt.ylabel('y(t)')
    plt.title('Example 1: dy/dt = -2y, y(0) = 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to file
    plt.savefig('tests/rk4_example1_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate error
    y_exact_at_points = analytical(t_num)
    error = np.abs(y_num - y_exact_at_points)
    print(f"Maximum error: {np.max(error):.2e}")
    
    # Save numerical results to CSV file
    results_data = np.column_stack((t_num, y_num, y_exact_at_points, error))
    np.savetxt('tests/rk4_example1_results.csv', results_data, 
               delimiter=',', 
               header='time,y_numerical,y_analytical,error',
               comments='')
    print("Results saved to 'rk4_example1_results.csv'")

# Example 2: Harmonic oscillator d²y/dt² + y = 0
# Convert to system: dy/dt = v, dv/dt = -y
def example2():
    def f(t, state):
        y, v = state
        dydt = v
        dvdt = -y
        return np.array([dydt, dvdt])
    
    # Initial conditions: y(0) = 1, v(0) = 0
    # Analytical solution: y = cos(t), v = -sin(t)
    def analytical(t):
        y = np.cos(t)
        v = -np.sin(t)
        return y, v
    
    t_span = (0, 4*np.pi)
    y0 = np.array([1, 0])  # [position, velocity]
    h = 0.01
    
    t_num, state_num = rk4_solve(f, t_span, y0, h)
    y_num = state_num[:, 0]  # position
    v_num = state_num[:, 1]  # velocity
    
    t_exact = np.linspace(0, 4*np.pi, 1000)
    y_exact, v_exact = analytical(t_exact)
    
    plt.figure(figsize=(12, 8))
    
    # Position plot
    plt.subplot(2, 2, 1)
    plt.plot(t_exact, y_exact, 'b-', label='Analytical', linewidth=2)
    plt.plot(t_num[::10], y_num[::10], 'ro', label='RK4', markersize=3)
    plt.xlabel('Time t')
    plt.ylabel('Position y(t)')
    plt.title('Harmonic Oscillator - Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Velocity plot
    plt.subplot(2, 2, 2)
    plt.plot(t_exact, v_exact, 'g-', label='Analytical', linewidth=2)
    plt.plot(t_num[::10], v_num[::10], 'ro', label='RK4', markersize=3)
    plt.xlabel('Time t')
    plt.ylabel('Velocity v(t)')
    plt.title('Harmonic Oscillator - Velocity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Phase portrait
    plt.subplot(2, 2, 3)
    plt.plot(y_exact, v_exact, 'b-', label='Analytical', linewidth=2)
    plt.plot(y_num, v_num, 'r--', label='RK4', linewidth=1)
    plt.xlabel('Position y')
    plt.ylabel('Velocity v')
    plt.title('Phase Portrait')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Energy conservation check
    energy_num = 0.5 * (y_num**2 + v_num**2)
    plt.subplot(2, 2, 4)
    plt.plot(t_num, energy_num, 'r-', linewidth=2)
    plt.axhline(y=0.5, color='b', linestyle='--', label='Theoretical energy = 0.5')
    plt.xlabel('Time t')
    plt.ylabel('Total Energy')
    plt.title('Energy Conservation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to file
    plt.savefig('tests/rk4_example2_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save numerical results to CSV file
    y_exact_at_points, v_exact_at_points = analytical(t_num)
    position_error = np.abs(y_num - y_exact_at_points)
    velocity_error = np.abs(v_num - v_exact_at_points)
    
    results_data = np.column_stack((t_num, y_num, v_num, 
                                   y_exact_at_points, v_exact_at_points,
                                   position_error, velocity_error, energy_num))
    np.savetxt('tests/rk4_example2_results.csv', results_data,
               delimiter=',',
               header='time,position_num,velocity_num,position_exact,velocity_exact,position_error,velocity_error,energy',
               comments='')
    
    print(f"Maximum position error: {np.max(position_error):.2e}")
    print(f"Maximum velocity error: {np.max(velocity_error):.2e}")
    print(f"Energy drift: {np.max(energy_num) - np.min(energy_num):.2e}")
    print("Results saved to 'rk4_example2_results.csv'")


if __name__ == "__main__":
    print("RK4 Method Examples")
    print("=" * 50)
    
    print("\nExample 1: Exponential decay")
    example1()
    
    print("\nExample 2: Harmonic oscillator")
    example2()