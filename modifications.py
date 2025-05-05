from sympy import symbols, integrate, Piecewise, And, simplify, sympify, lambdify, exp, sin, cos, Heaviside, DiracDelta, log
from sympy.abc import t, tau
import numpy as np
import matplotlib.pyplot as plt
import warnings

def rectangular_convolution_analysis():
    """
    Comprehensive analysis of convolution with rectangular kernels:
    1. Symmetric rectangular kernel h(t) = 1 for -T ≤ t ≤ T, 0 otherwise
    2. Asymmetric rectangular kernel h(t) = 1 for t > 0, 0 otherwise
    3. Time-shifted rectangular kernel h(t-τ₀)
    
    Displays all results in a side-by-side figure.
    """
    print("\n=== Convolution Analysis with Rectangular Kernel ===")
    
    # Get user input or use default values
    try:
        print("\nEnter the input signal f(t) (examples: sin(t), t**2, exp(-t), Heaviside(t), log(abs(t)+1)):")
        f_input = input("f(t) = ")
        if not f_input.strip():
            f_input = "Heaviside(t)"  # Default
            print(f"Using default: f(t) = {f_input}")
            
        print("\nEnter the width T of the rectangular kernel (e.g., 0.5, 1):")
        T_input = input("T = ")
        if not T_input.strip():
            T_input = "1"  # Default
            print(f"Using default: T = {T_input}")
            
        print("\nEnter time shift τ₀ for the shifted kernel (e.g., 0.5, 1):")
        tau0_input = input("τ₀ = ")
        if not tau0_input.strip():
            tau0_input = "0.5"  # Default
            print(f"Using default: τ₀ = {tau0_input}")
            
        # Parse inputs
        f_expr = sympify(f_input)
        T_val = float(T_input)
        tau0_val = float(tau0_input)
        
    except Exception as e:
        print(f"Error with input: {e}")
        print("Using default values: f(t) = Heaviside(t), T = 1, τ₀ = 0.5")
        f_expr = Heaviside(t)
        T_val = 1.0
        tau0_val = 0.5
    
    # Define the kernels
    h_sym = lambda x: 1 if -T_val <= x <= T_val else 0        # Symmetric
    h_asym = lambda x: 1 if 0 <= x <= T_val else 0            # Asymmetric (t>0)
    h_shift = lambda x: 1 if -T_val+tau0_val <= x <= T_val+tau0_val else 0  # Time-shifted
    
    # Compute convolutions numerically for plotting
    t_vals = np.linspace(-5, 5, 1000)
    
    # Evaluate the input signal f(t)
    f_func = lambdify(t, f_expr, modules=['numpy', {'Heaviside': lambda x: np.heaviside(x, 0.5),
                                                   'DiracDelta': lambda x: np.zeros_like(x),
                                                   'log': lambda x: np.log(np.abs(x) + 1e-10)}])
    try:
        f_vals = f_func(t_vals)
    except Exception as e:
        print(f"Error evaluating function: {e}")
        f_vals = np.zeros_like(t_vals)
        for i, t_val in enumerate(t_vals):
            try:
                f_vals[i] = float(f_expr.subs(t, t_val))
            except:
                f_vals[i] = np.nan
    
    # Compute numerical convolution for each kernel
    dt = t_vals[1] - t_vals[0]
    tau_vals = t_vals.copy()
    
    y_sym = np.zeros_like(t_vals)
    y_asym = np.zeros_like(t_vals)
    y_shift = np.zeros_like(t_vals)
    
    for i, t_val in enumerate(t_vals):
        # For each t, compute the convolution integral numerically
        for j, tau_val in enumerate(tau_vals):
            if t_val - tau_val < t_vals[0] or t_val - tau_val > t_vals[-1]:
                continue
                
            # Find the closest index for t-tau
            idx = np.abs(t_vals - (t_val - tau_val)).argmin()
            
            # Apply the kernels at t-tau
            h_sym_val = h_sym(t_val - tau_val)
            h_asym_val = h_asym(t_val - tau_val)
            h_shift_val = h_shift(t_val - tau_val)
            
            # Accumulate the integrals
            if not np.isnan(f_vals[j]):
                y_sym[i] += f_vals[j] * h_sym_val * dt
                y_asym[i] += f_vals[j] * h_asym_val * dt
                y_shift[i] += f_vals[j] * h_shift_val * dt
    
    # Create plot
    fig = plt.figure(figsize=(15, 10))
    
    # LEFT SIDE: Symmetric vs Asymmetric Analysis
    # Plot input signal
    plt.subplot(3, 2, 1)
    plt.plot(t_vals, f_vals, label=f'$f(t)$', color='blue')
    plt.title('Input Signal')
    plt.grid(True)
    plt.legend()
    
    # Plot symmetric and asymmetric kernels
    plt.subplot(3, 2, 3)
    h_sym_vals = np.array([h_sym(x) for x in t_vals])
    h_asym_vals = np.array([h_asym(x) for x in t_vals])
    plt.plot(t_vals, h_sym_vals, label='Symmetric Kernel', color='green')
    plt.plot(t_vals, h_asym_vals, label='Asymmetric Kernel (t>0)', color='purple')
    plt.title('Kernel Comparison')
    plt.grid(True)
    plt.legend()
    
    # Plot symmetric and asymmetric convolution results
    plt.subplot(3, 2, 5)
    plt.plot(t_vals, y_sym, label='Symmetric Convolution', color='red')
    plt.plot(t_vals, y_asym, label='Asymmetric Convolution', color='orange')
    plt.title('Convolution Results')
    plt.grid(True)
    plt.legend()
    
    # RIGHT SIDE: Time-Shifted Analysis
    # Plot input signal again for reference
    plt.subplot(3, 2, 2)
    plt.plot(t_vals, f_vals, label=f'$f(t)$', color='blue')
    plt.title('Input Signal')
    plt.grid(True)
    plt.legend()
    
    # Plot original and shifted kernels
    plt.subplot(3, 2, 4)
    h_shift_vals = np.array([h_shift(x) for x in t_vals])
    plt.plot(t_vals, h_sym_vals, label='Original Kernel', color='green')
    plt.plot(t_vals, h_shift_vals, label=f'Shifted Kernel (τ₀={tau0_val})', color='purple')
    plt.title('Kernel Shift Comparison')
    plt.grid(True)
    plt.legend()
    
    # Plot original and shifted convolution results
    plt.subplot(3, 2, 6)
    plt.plot(t_vals, y_sym, label='Original Convolution', color='red')
    plt.plot(t_vals, y_shift, label='Shifted Convolution', color='magenta')
    plt.title('Convolution with Time Shift')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Add section for logarithmic function analysis if log function is used
    if 'log' in str(f_expr):
        print("\n=== Special Analysis for Logarithmic Input ===")
        print("The logarithmic function has important properties in convolution analysis:")
        print("1. For t near zero, the logarithm approaches negative infinity, which can cause")
        print("   numerical instabilities in the convolution integral.")
        print("2. The shape of the logarithmic function (slow growth) creates a smoothing effect")
        print("   that is different from polynomial or exponential inputs.")
        print("3. When using log(abs(t)+c), the parameter c controls how the singularity at t=0")
        print("   is handled, affecting the overall convolution result.")
    
    # Print analytical explanation
    print("\nAnalytical Results:")
    print("\n1. The convolution with a symmetric rectangular kernel acts as a moving average filter")
    print("   over a window of width 2T centered at the current time t.")
    
    print("\n2. When modified to only consider t > 0 (asymmetric kernel), the convolution")
    print("   becomes a causal filter that only averages past values, which is physically realizable.")
    
    print("\n3. Shifting the kernel by τ₀ shifts the convolution result by the same amount.")
    print("   This illustrates the time-delay property of convolution:")
    print("   If y(t) = f(t) * h(t), then f(t) * h(t-τ₀) = y(t-τ₀)")
    print("   In the context of systems, this represents a time delay of τ₀ in the system response.")

if __name__ == "__main__":
    rectangular_convolution_analysis()
