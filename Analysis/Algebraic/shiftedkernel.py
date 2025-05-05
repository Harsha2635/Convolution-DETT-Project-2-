import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2, 2, 1000)
dt = t[1] - t[0]

# Define shifted kernel parameters
T = 0.2      # Width of rectangular pulse
τ0 = 0.1    # Time shift
kernel_t = np.arange(-T, T + dt, dt)
h = np.ones_like(kernel_t)

# Shift kernel: pad with zeros to simulate h(t - τ0)
shift_samples = int(τ0 / dt)
h_shifted = np.pad(h, (shift_samples, 0), mode='constant')[:len(h)]
h_shifted = h_shifted / np.sum(h_shifted)  # Normalize

# Define a list of exponents for t^n
powers = [1, 2, 3, -1, 0.5]

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
font = {'family': 'serif', 'size': 12}
plt.rc('font', **font)

# Convolution for each function t^n
for n in powers:
    # Avoid domain errors for negative or undefined values
    func = np.where((t != 0) | (n > 0), np.power(t, n, where=(t != 0) | (n > 0)), 0)
    func = np.nan_to_num(func, nan=0.0, posinf=0.0, neginf=0.0)

    # Convolve with shifted kernel
    conv_result = convolve(func, h_shifted, mode='same')

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, func, label=f'$t^{{{n}}}$', linestyle='--', linewidth=2, color='tab:gray')
    ax.plot(t, conv_result, label='Shifted Convolution', linewidth=2, color='tab:blue')

    ax.set_title(f'Convolution of $t^{n}$ with Shifted Kernel $h(t - \\tau_0)$,  $\\tau_0 = {τ0}$', fontsize=14)
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=11)

    plt.tight_layout()
    plt.show()

