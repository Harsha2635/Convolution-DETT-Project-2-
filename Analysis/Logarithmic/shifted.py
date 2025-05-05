import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2, 2, 1000)
dt = t[1] - t[0]

# Define the logarithmic function
def log_function(x):
    return np.log(1 + np.abs(x))

f = log_function(t)

# Define the original rectangular kernel width T and shift τ0
T = 0.2
τ0 = 0.15  # time‐shift of the kernel

# Build symmetric kernel h(t) on [–T, +T]
kernel_t = np.arange(-T, T + dt, dt)
h = np.ones_like(kernel_t)

# Shift h by τ0: pad on left by shift_samples, then truncate/pad to original length
shift_samples = int(np.round(τ0 / dt))
h_shifted = np.pad(h, (shift_samples, 0), mode='constant')[: len(h)]

# Normalize so area(h_shifted) = 1
h_shifted /= h_shifted.sum()

# Convolve with shifted kernel
f_conv_shifted = convolve(f, h_shifted, mode='same')

# Plot original and shifted‐convolved
plt.figure(figsize=(8, 4))
plt.plot(t, f,           linestyle='--', label='$f(t)=\\ln(1+|t|)$')
plt.plot(t, f_conv_shifted,
         label=f'Convolution with $h(t-\\tau_0)$, $\\tau_0={τ0}$',
         color='C2')
plt.title('Convolution of $\\ln(1+|t|)$ with Shifted Rectangular Kernel')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.tight_layout()
plt.show()

