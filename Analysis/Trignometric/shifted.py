import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2*np.pi, 2*np.pi, 1000)
dt = t[1] - t[0]

# Trigonometric functions
sin_t = np.sin(t)
cos_t = np.cos(t)
tan_t = np.tan(np.clip(t, -np.pi/2+0.1, np.pi/2-0.1))

# Shifted kernel parameters
T = 0.5       # Kernel width
tau = 0.3     # Shift amount

# Create shifted kernel [τ, τ+T]
kernel_shift_t = np.arange(tau, tau+T+dt, dt)
h_shift = np.ones_like(kernel_shift_t)
h_shift /= h_shift.sum()  # Normalize

# Pad kernel to match signal length
h_shift_padded = np.zeros_like(t)
start_idx = np.argmin(np.abs(t - tau))
h_shift_padded[start_idx:start_idx+len(kernel_shift_t)] = h_shift

# Perform convolution
sin_conv_shift = convolve(sin_t, h_shift_padded, mode='same')
cos_conv_shift = convolve(cos_t, h_shift_padded, mode='same')
tan_conv_shift = convolve(tan_t, h_shift_padded, mode='same')

# Plotting function
def plot_shifted(func, conv, name, color):
    plt.figure(figsize=(8, 4))
    plt.plot(t, func, '--', label=f'{name}(t)')
    plt.plot(t, conv, label=f'Shifted conv (τ={tau})', color=color)
    plt.title(f'Convolution of {name}(t) with shifted kernel [τ,τ+T]')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_conv_shifted.png')
    plt.show()

# Generate plots
plot_shifted(sin_t, sin_conv_shift, 'sin', 'blue')
plot_shifted(cos_t, cos_conv_shift, 'cos', 'green')
plot_shifted(tan_t, tan_conv_shift, 'tan', 'red')
