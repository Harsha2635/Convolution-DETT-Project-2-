import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2*np.pi, 2*np.pi, 1000)  # Extended range for periodic functions
dt = t[1] - t[0]

# Define the trigonometric functions
sin_t = np.sin(t)
cos_t = np.cos(t)
tan_t = np.tan(np.clip(t, -np.pi/2+0.1, np.pi/2-0.1))  # Avoid asymptotes

# Define the modified kernel (t > 0 only)
T = 0.5  # Kernel width
kernel_t = np.arange(0, T+dt, dt)  # Only positive times
h_mod = np.ones_like(kernel_t)
h_mod /= h_mod.sum()  # Normalize

# Pad kernel to match signal length
h_mod_padded = np.zeros_like(t)
h_mod_padded[:len(kernel_t)] = h_mod

# Perform convolution
sin_conv_mod = convolve(sin_t, h_mod_padded, mode='same')
cos_conv_mod = convolve(cos_t, h_mod_padded, mode='same')
tan_conv_mod = convolve(tan_t, h_mod_padded, mode='same')

# Plotting for sin(t) with modified kernel
plt.figure(figsize=(8, 4))
plt.plot(t, sin_t, '--', label='sin(t)')
plt.plot(t, sin_conv_mod, label='Convolved (t>0 kernel)', color='blue')
plt.title('Convolution of sin(t) with modified kernel (t > 0)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-2*np.pi, 2*np.pi)
plt.tight_layout()
plt.savefig('sin_conv_modified.png')
plt.show()

# Plotting for cos(t) with modified kernel
plt.figure(figsize=(8, 4))
plt.plot(t, cos_t, '--', label='cos(t)')
plt.plot(t, cos_conv_mod, label='Convolved (t>0 kernel)', color='green')
plt.title('Convolution of cos(t) with modified kernel (t > 0)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-2*np.pi, 2*np.pi)
plt.tight_layout()
plt.savefig('cos_conv_modified.png')
plt.show()

# Plotting for tan(t) with modified kernel
plt.figure(figsize=(8, 4))
plt.plot(t, tan_t, '--', label='tan(t)')
plt.plot(t, tan_conv_mod, label='Convolved (t>0 kernel)', color='red')
plt.title('Convolution of tan(t) with modified kernel (t > 0)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-2*np.pi, 2*np.pi)
plt.tight_layout()
plt.savefig('tan_conv_modified.png')
plt.show()
