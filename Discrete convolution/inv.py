import numpy as np
import matplotlib.pyplot as plt

# Define discrete time range
n_min, n_max = -50, 50  # Reduced range for computational efficiency
n = np.arange(n_min, n_max)

# Define rectangular kernel
def rect_kernel(n, N):
    return np.where((n >= -N) & (n <= N), 1, 0)

# Inverse Trig function: arctan(n)
def inverse_trig_function(n):
    return np.arctan(n/10)  # Scaled to avoid steep transitions

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
f_values = inverse_trig_function(n)
h_values = rect_kernel(n, N)

# Perform manual convolution
print("Computing manual convolution for inverse trig function...")
y_values = manual_discrete_convolution(f_values, h_values, n)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(n, f_values)
plt.title('Input Signal: f[n] = arctan(n/10)')
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

# Alternative implementation using direct sum within kernel width
def direct_discrete_convolution(f_values, h_values, n_range, N):
    result = np.zeros(len(n_range))
    
    for i, n_val in enumerate(n_range):
        # For rectangular kernel, we can directly sum values within the kernel width
        # This is more efficient for this specific kernel
        lower_idx = max(0, i - N)
        upper_idx = min(len(n_range), i + N + 1)
        result[i] = np.sum(f_values[lower_idx:upper_idx])
    
    return result

# Compare with direct method
print("Computing direct convolution for inverse trig function...")
y_direct = direct_discrete_convolution(f_values, h_values, n, N)

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
plt.title('Input Signal: f[n] = arctan(n/10)')
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
tau_0 = 3  # Shift value (smaller for visual clarity)
h_shifted = np.roll(h_values, tau_0)
print("Computing shifted kernel convolution...")
y_shifted = manual_discrete_convolution(f_values, h_shifted, n)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(n, f_values)
plt.title('Input Signal: f[n] = arctan(n/10)')
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

print("Inverse Trigonometric Function Manual Convolution Analysis:")
print(f"- Original kernel width parameter N = {N}")
print(f"- Shift parameter tau_0 = {tau_0}")
print("- The rectangular kernel performs a moving average operation")
print("- Half-sided kernel causes asymmetry in the output")
print("- Shifting the kernel by tau_0 shifts the output signal")
