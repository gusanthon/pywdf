import math

def newton_raphson(f, df, x0, tolerance=1e-10, max_iterations=100):
    """
    Find the root of a function using the Newton-Raphson method.
    
    Parameters:
    f: function whose root we want to find
    df: derivative of the function f
    x0: initial guess
    tolerance: convergence criterion (default: 1e-10)
    max_iterations: maximum number of iterations (default: 100)
    
    Returns:
    root: approximate root of the function
    iterations: number of iterations performed
    """
    x = x0
    
    for i in range(max_iterations):
        fx = f(x)
        dfx = df(x)
        
        # Check if derivative is too close to zero
        if abs(dfx) < 1e-15:
            raise ValueError(f"Derivative too close to zero at x = {x}")
        
        # Newton-Raphson formula: x_new = x - f(x)/f'(x)
        x_new = x - fx / dfx
        
        # Check for convergence
        if abs(x_new - x) < tolerance:
            return x_new, i + 1
        
        x = x_new
    
    raise ValueError(f"Failed to converge after {max_iterations} iterations")

