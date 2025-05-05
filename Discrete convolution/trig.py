import numpy as np
import matplotlib.pyplot as plt

# Define discrete time range
n_min, n_max = -50, 50  # Reduced range for computational efficiency
n = np.arange(n_min, n_max)

# Define rectangular kernel
def rect_kernel(n, N):
    return np.where((n >= -N) & (n <= N), 1, 0)

# Logarithmic function: log(|n| + 1)
def log_function(n):
    return np.log(np.abs(n) + 1)

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
f_values = log_function(n)
h_values = rect_kernel(n, N)

# Perform manual convolution
print("Computing manual convolution for logarithmic function...")
y_values = manual_discrete_convolution(f_values, h_values, n)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(n, f_values)
plt.title('Input Signal: f[n] = log(|n| + 1)')
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

# Optimize
