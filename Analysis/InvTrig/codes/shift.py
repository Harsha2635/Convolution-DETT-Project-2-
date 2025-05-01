import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2, 2, 1000)

# Define the inverse trigonometric functions
sin_inv = np.arcsin(np.clip(t, -1, 1))   # arcsin is defined only for -1 <= t <= 1
cos_inv = np.arccos(np.clip(t, -1, 1))   # arccos is defined only for -1 <= t <= 1
tan_inv = np.arctan(t)                   # arctan is defined for all real numbers

# Define the original kernel h(t) with a shift τ0
T = 0.2
τ0 = 0.15  # shift in time

# Kernel shifted by τ0: center moves right by τ0
dt = t[1] - t[0]
kernel_t = np.arange(-T, T + dt, dt)
h = np.ones_like(kernel_t)

# Apply shift by zero-padding from the left
shift_samples = int(τ0 / dt)
h_shifted = np.pad(h, (shift_samples, 0), mode='constant')[:len(h)]

# Normalize the shifted kernel
h_shifted = h_shifted / np.sum(h_shifted)

# Perform convolution with shifted kernel
sin_conv = convolve(sin_inv, h_shifted, mode='same')
cos_conv = convolve(cos_inv, h_shifted, mode='same')
tan_conv = convolve(tan_inv, h_shifted, mode='same')

# Plotting for arcsin(t)
plt.figure(figsize=(8, 4))
plt.plot(t, sin_inv, label='arcsin(t)', linestyle='--')
plt.plot(t, sin_conv, label='Shifted Convolution', color='blue')
plt.title(f'Convolution of arcsin(t) with Shifted h(t - τ₀), τ₀ = {τ0}')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figs/shift1.png')
plt.show()

# Plotting for arccos(t)
plt.figure(figsize=(8, 4))
plt.plot(t, cos_inv, label='arccos(t)', linestyle='--')
plt.plot(t, cos_conv, label='Shifted Convolution', color='green')
plt.title(f'Convolution of arccos(t) with Shifted h(t - τ₀), τ₀ = {τ0}')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figs/shift2.png')
plt.show()

# Plotting for arctan(t)
plt.figure(figsize=(8, 4))
plt.plot(t, tan_inv, label='arctan(t)', linestyle='--')
plt.plot(t, tan_conv, label='Shifted Convolution', color='red')
plt.title(f'Convolution of arctan(t) with Shifted h(t - τ₀), τ₀ = {τ0}')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figs/shift3.png')
plt.show()

