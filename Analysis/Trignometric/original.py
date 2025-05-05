import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2*np.pi, 2*np.pi, 1000)  # Extended range for periodic functions

# Define the trigonometric functions
sin_t = np.sin(t)
cos_t = np.cos(t)
tan_t = np.tan(np.clip(t, -np.pi/2+0.1, np.pi/2-0.1))  # Avoid asymptotes

# Define the kernel h(t), a rectangular pulse from -T to T
T = 0.5  # Kernel width
dt = t[1] - t[0]
kernel_t = np.arange(-T, T+dt, dt)
h = np.ones_like(kernel_t)
h /= h.sum()  # Normalize

# Perform convolution
sin_conv = convolve(sin_t, h, mode='same')
cos_conv = convolve(cos_t, h, mode='same')
tan_conv = convolve(tan_t, h, mode='same')

# Plotting for sin(t)
plt.figure(figsize=(8, 4))
plt.plot(t, sin_t, label='sin(t)', linestyle='--')
plt.plot(t, sin_conv, label='Convolved', color='blue')
plt.title('Convolution of sin(t) with symmetric kernel')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-2*np.pi, 2*np.pi)
plt.tight_layout()
plt.savefig('sin_conv_original.png')
plt.show()

# Plotting for cos(t)
plt.figure(figsize=(8, 4))
plt.plot(t, cos_t, label='cos(t)', linestyle='--')
plt.plot(t, cos_conv, label='Convolved', color='green')
plt.title('Convolution of cos(t) with symmetric kernel')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-2*np.pi, 2*np.pi)
plt.tight_layout()
plt.savefig('cos_conv_original.png')
plt.show()

# Plotting for tan(t)
plt.figure(figsize=(8, 4))
plt.plot(t, tan_t, label='tan(t)', linestyle='--')
plt.plot(t, tan_conv, label='Convolved', color='red')
plt.title('Convolution of tan(t) with symmetric kernel')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-2*np.pi, 2*np.pi)
plt.tight_layout()
plt.savefig('tan_conv_original.png')
plt.show()
