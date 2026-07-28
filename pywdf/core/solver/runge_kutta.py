import numpy as np

def rk4_step(f, t, y, h):
    """
    Single step of fourth-order Runge-Kutta method
    
    Parameters:
    f: function that defines dy/dt = f(t, y)
    t: current time
    y: current value of y
    h: step size
    
    Returns:
    y_next: next value of y
    """
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    
    y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_next

def rk4_solve(f, t_span, y0, h):
    """
    Solve ODE using RK4 method
    
    Parameters:
    f: function that defines dy/dt = f(t, y)
    t_span: tuple (t_start, t_end)
    y0: initial condition (scalar or array)
    h: step size
    
    Returns:
    t_values: array of time points
    y_values: array of solution values
    """
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + h, h)
    
    # Handle both scalar and vector initial conditions
    y0 = np.asarray(y0)
    if y0.ndim == 0:  # Scalar case
        y_values = np.zeros(len(t_values))
    else:  # Vector case
        y_values = np.zeros((len(t_values), len(y0)))
    
    # Initial condition
    y_values[0] = y0
    
    # Apply RK4 method
    for i in range(len(t_values) - 1):
        y_values[i + 1] = rk4_step(f, t_values[i], y_values[i], h)
    
    return t_values, y_values


