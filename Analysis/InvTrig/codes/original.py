import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2, 2, 1000)

# Define the inverse trigonometric functions
sin_inv = np.arcsin(np.clip(t, -1, 1))   # arcsin is defined only for -1 <= t <= 1
cos_inv = np.arccos(np.clip(t, -1, 1))   # arccos is defined only for -1 <= t <= 1
tan_inv = np.arctan(t)                   # arctan is defined for all real numbers

# Define the kernel h(t), a rectangular pulse from -T to T
T = 0.2
dt = t[1] - t[0]
kernel_t = np.arange(-T, T+dt, dt)
h = np.ones_like(kernel_t)

# Normalize the kernel if needed (e.g., for smoothing)
h /= h.sum()

# Perform convolution
sin_conv = convolve(sin_inv, h, mode='same')
cos_conv = convolve(cos_inv, h, mode='same')
tan_conv = convolve(tan_inv, h, mode='same')

# Plotting for arcsin(t)
plt.figure(figsize=(8, 4))
plt.plot(t, sin_inv, label='arcsin(t)', linestyle='--')
plt.plot(t, sin_conv, label='Convolved', color='blue')
plt.title('Convolution of arcsin(t) with h(t)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figs/org1.png')
plt.show()

# Plotting for arccos(t)
plt.figure(figsize=(8, 4))
plt.plot(t, cos_inv, label='arccos(t)', linestyle='--')
plt.plot(t, cos_conv, label='Convolved', color='green')
plt.title('Convolution of arccos(t) with h(t)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figs/org2.png')
plt.show()

# Plotting for arctan(t)
plt.figure(figsize=(8, 4))
plt.plot(t, tan_inv, label='arctan(t)', linestyle='--')
plt.plot(t, tan_conv, label='Convolved', color='red')
plt.title('Convolution of arctan(t) with h(t)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figs/org3.png')
plt.show()
