import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import os  # For directory creation

# Time axis
t = np.linspace(-2, 2, 1000)
dt = t[1] - t[0]

# Step function u(t)
step = np.heaviside(t, 1)

# Delta function approximation (narrow Gaussian spike)
delta = np.zeros_like(t)
delta[np.abs(t) < 0.01] = 1 / dt  # Area ≈ 1

# Define kernels
T = 0.2
τ0 = 0.15

# Causal kernel: h(t) = 1 on [0, T]
kernel_t_causal = np.arange(0, T + dt, dt)
h_causal = np.ones_like(kernel_t_causal)
h_causal /= h_causal.sum()

# Symmetric kernel: h(t) = 1 on [-T, T]
kernel_t_sym = np.arange(-T, T + dt, dt)
h_sym = np.ones_like(kernel_t_sym)
h_sym /= h_sym.sum()

# Shifted kernel: shift symmetric kernel by τ0
shift_samples = int(τ0 / dt)
h_shifted = np.pad(h_sym, (shift_samples, 0), mode='constant')[:len(h_sym)]
h_shifted /= h_shifted.sum()

# Create directory for saving figures
output_dir = '../figs/'
os.makedirs(output_dir, exist_ok=True)

def plot_conv(signal, kernel, label, filename):
    conv_result = convolve(signal, kernel, mode='same')
    plt.figure(figsize=(8, 4))
    plt.plot(t, signal, '--', label='Input Signal')
    plt.plot(t, conv_result, label=label, color='purple')
    plt.title(f'Convolution: Input * {label}')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}{filename}.png')  # Save to the created directory
    plt.show()

# ---- Step Convolutions ----
plot_conv(step, h_causal, 'Causal Kernel', 'step_causal')
plot_conv(step, h_sym, 'Symmetric Kernel', 'step_symmetric')
plot_conv(step, h_shifted, f'Shifted Kernel (τ₀={τ0})', 'step_shifted')

# ---- Delta Convolutions ----
plot_conv(delta, h_causal, 'Causal Kernel', 'delta_causal')
plot_conv(delta, h_sym, 'Symmetric Kernel', 'delta_symmetric')
plot_conv(delta, h_shifted, f'Shifted Kernel (τ₀={τ0})', 'delta_shifted')

