from sympy import symbols, integrate, Piecewise, And, simplify, sympify, lambdify
from sympy.abc import t, tau
import numpy as np
import matplotlib.pyplot as plt
import warnings

def rectangular_convolution_with_plot():
    """
    Working version that always shows output and plots
    """
    print("\nEnter the input signal f(t) (examples: sin(t), asin(t), t**2, exp(-t)):")
    f_input = input("f(t) = ")
    
    print("\nEnter the width T of the rectangular kernel (e.g., 0.5, 1):")
    T_input = input("T = ")
    
    # Parse inputs
    try:
        f_expr = sympify(f_input)
        T_val = float(T_input)
        T = symbols('T', real=True, positive=True)
    except:
        print("Invalid input! Please use valid mathematical expressions.")
        return
    
    # Define rectangular kernel
    h_tau = Piecewise((1, And(-T <= tau, tau <= T)), (0, True))
    
    # Compute convolution
    h_t_tau = h_tau.subs(tau, t - tau)
    lower_limit = t - T
    upper_limit = t + T
    integrand = f_expr.subs(t, tau) * h_t_tau
    
    # Compute integral
    try:
        y_expr = simplify(integrate(integrand, (tau, lower_limit, upper_limit)))
        y_expr = y_expr.subs(T, T_val)
        print("\nConvolution Result:")
        print(f"y(t) = {y_expr}")
    except:
        print("Could not compute the integral symbolically. Trying numerical evaluation.")
        y_expr = None

    # Plotting parameters
    t_min = -3 if y_expr is None else max(-3, -1 - T_val)  # Wider range if no symbolic result
    t_max = 3 if y_expr is None else min(3, 1 + T_val)
    t_vals = np.linspace(t_min, t_max, 1000)
    
    # Evaluation functions
    def safe_eval(expr, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return float(expr.subs(t, x))
            except:
                return np.nan
    
    # Evaluate functions
    f_vals = np.array([safe_eval(f_expr, x) for x in t_vals])
    h_vals = np.array([1 if -T_val <= x <= T_val else 0 for x in t_vals])
    
    if y_expr is not None:
        y_vals = np.array([safe_eval(y_expr, x) for x in t_vals])
    else:
        # Numerical convolution fallback
        dt = t_vals[1] - t_vals[0]
        y_vals = np.convolve(
            [safe_eval(f_expr, x) for x in t_vals],
            [1 if -T_val <= x <= T_val else 0 for x in t_vals],
            mode='same'
        ) * dt

    # Create plots
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_vals, f_vals, label=f'$f(t) = {str(f_expr).replace("**", "^")}$', color='blue')
    plt.title('Input Signal')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(t_vals, h_vals, label=f'$h(t)$ (T={T_val})', color='green')
    plt.title('Rectangular Kernel')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t_vals, y_vals, label='$y(t) = (f * h)(t)$', color='red')
    plt.title('Convolution Result')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Run the function
print("=== Convolution Calculator ===")
rectangular_convolution_with_plot()
