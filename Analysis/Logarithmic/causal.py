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

# Define a causal kernel h(t) for t > 0
T = 0.2
kernel_t = np.arange(0, T + dt, dt)
h = np.ones_like(kernel_t)
h /= h.sum()      # normalize so area = 1

# Perform causal convolution
f_conv = convolve(f, h, mode='same')

# Plot original vs. convolved
plt.figure(figsize=(8, 4))
plt.plot(t, f,    linestyle='--', label='$f(t)=\\ln(1+|t|)$')
plt.plot(t, f_conv, label='Causal Convolution', color='C1')
plt.title('Causal Convolution of $\\ln(1+|t|)$ with $h(t>0)$')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.tight_layout()
plt.show()

