import numpy as np
import matplotlib.pyplot as plt

# Define discrete time range
n_min, n_max = -50, 50  # Reduced range for computational efficiency
n = np.arange(n_min, n_max)

# Define rectangular kernel
def rect_kernel(n, N):
    return np.where((n >= -N) & (n <= N), 1, 0)

# Algebraic function: n²
def algebraic_function(n):
    return n**2

# Manual implementation of discrete convolution
def manual_discrete_convolution(f_values, h_values, n_range):
    result = np.zeros(len(n_range))
    
    for i, n_val in enumerate(n_range):
        sum_val = 0
        for m, m_val in enumerate(n_range):
            # Find corresponding h[n-m] index
            h_idx = n_val - m_val
            # Check if h_idx is within range
            if h_idx >= n_min and h_idx < n_max:
                h_idx_adjusted = h_idx - n_min  # Adjust index for array access
                sum_val += f_values[m] * h_values[h_idx_adjusted]
        
        result[i] = sum_val
    
    return result

# Parameters
N = 5  # Half-width of the rectangular kernel (smaller N for faster computation)

# Calculate values
f_values = algebraic_function(n)
h_values = rect_kernel(n, N)

# Perform manual convolution
print("Computing manual convolution for algebraic function...")
y_values = manual_discrete_convolution(f_values, h_values, n)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(n, f_values)
plt.title('Input Signal: f[n] = n²')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(n, h_values)
plt.title(f'Rectangular Kernel: h[n], N = {N}')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(n, y_values)
plt.title('Manual Convolution Result: y[n] = (f * h)[n]')
plt.grid(True)

plt.tight_layout()
plt.show()

# Alternative implementation for rectangular kernel
def direct_discrete_convolution(f_values, n_range, N):
    result = np.zeros(len(n_range))
    
    for i, n_val in enumerate(n_range):
        # For rectangular kernel, directly sum values within kernel width
        sum_val = 0
        for k in range(-N, N+1):
            idx = i + k
            if 0 <= idx < len(n_range):
                sum_val += f_values[idx]
        result[i] = sum_val
    
    return result

# Compare with direct method
print("Computing direct convolution for algebraic function...")
y_direct = direct_discrete_convolution(f_values, n, N)

plt.figure(figsize=(10, 6))
plt.plot(n, y_values, 'b-', label='Manual Convolution')
plt.plot(n, y_direct, 'r--', label='Direct Sum Method')
plt.title('Comparison of Convolution Methods')
plt.legend()
plt.grid(True)
plt.show()

# Part (a): Half-sided kernel
h_half_values = np.where((n >= 0) & (n <= N), 1, 0)
print("Computing half-sided kernel convolution...")
y_half_values = manual_discrete_convolution(f_values, h_half_values, n)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(n, f_values)
plt.title('Input Signal: f[n] = n²')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(n, h_half_values)
plt.title('Half-sided Kernel: h[n], n >= 0')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(n, y_half_values)
plt.title('Manual Convolution with Half-sided Kernel')
plt.grid(True)

plt.tight_layout()
plt.show()

# Part (b): Shifted kernel
tau_0 = 3  # Shift value
h_shifted = np.roll(h_values, tau_0)
print("Computing shifted kernel convolution...")
y_shifted = manual_discrete_convolution(f_values, h_shifted, n)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(n, f_values)
plt.title('Input Signal: f[n] = n²')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(n, h_shifted)
plt.title(f'Shifted Kernel: h[n-{tau_0}]')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(n, y_shifted)
plt.title('Manual Convolution with Shifted Kernel')
plt.grid(True)

plt.tight_layout()
plt.show()

# Analytical analysis for rectangular kernel with quadratic function
# For f[n] = n² and rectangular kernel width 2N+1, we expect:
# y[n] = (2N+1)n² + N(N+1)(2N+1)/3
n_analytical = np.linspace(-20, 20, 41)
analytical_result = (2*N+1)*n_analytical**2 + N*(N+1)*(2*N+1)/3

plt.figure(figsize=(10, 6))
plt.plot(n, y_values, 'b-', label='Numerical Convolution')
plt.plot(n_analytical, analytical_result, 'r--', label='Analytical: (2N+1)n² + N(N+1)(2N+1)/3')
plt.title('Comparison with Analytical Result for Quadratic Function')
plt.legend()
plt.grid(True)
plt.show()

print("Algebraic Function Manual Convolution Analysis:")
print(f"- Original kernel width parameter N = {N}")
print(f"- Shift parameter tau_0 = {tau_0}")
print("- For quadratic function f[n] = n², the convolution scales the function and adds a constant")
print("- The analytical expression for rectangular kernel convolution is: (2N+1)n² + N(N+1)(2N+1)/3")
print("- Half-sided kernel creates asymmetric averaging")
print("- Shifting the kernel shifts the output signal")
