import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2, 2, 1000)

# Define the inverse trigonometric functions
sin_inv = np.arcsin(np.clip(t, -1, 1))   # arcsin is defined only for -1 <= t <= 1
cos_inv = np.arccos(np.clip(t, -1, 1))   # arccos is defined only for -1 <= t <= 1
tan_inv = np.arctan(t)                   # arctan is defined for all real numbers

# Define a causal kernel h(t) for t > 0
T = 0.2
dt = t[1] - t[0]
kernel_t = np.arange(0, T + dt, dt)
h = np.ones_like(kernel_t)

# Normalize the kernel
h /= h.sum()

# Perform convolution (causal)
sin_conv = convolve(sin_inv, h, mode='same')
cos_conv = convolve(cos_inv, h, mode='same')
tan_conv = convolve(tan_inv, h, mode='same')

# Plotting for arcsin(t)
plt.figure(figsize=(8, 4))
plt.plot(t, sin_inv, label='arcsin(t)', linestyle='--')
plt.plot(t, sin_conv, label='Causal Convolution', color='blue')
plt.title('Causal Convolution of arcsin(t) with h(t > 0)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figs/mod1.png')
plt.show()

# Plotting for arccos(t)
plt.figure(figsize=(8, 4))
plt.plot(t, cos_inv, label='arccos(t)', linestyle='--')
plt.plot(t, cos_conv, label='Causal Convolution', color='green')
plt.title('Causal Convolution of arccos(t) with h(t > 0)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figs/mod2.png')
plt.show()

# Plotting for arctan(t)
plt.figure(figsize=(8, 4))
plt.plot(t, tan_inv, label='arctan(t)', linestyle='--')
plt.plot(t, tan_conv, label='Causal Convolution', color='red')
plt.title('Causal Convolution of arctan(t) with h(t > 0)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figs/mod3.png')
plt.show()
