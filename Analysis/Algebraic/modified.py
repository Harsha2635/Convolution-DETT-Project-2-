import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2, 2, 1000)
dt = t[1] - t[0]

# Define causal kernel h(t) as a rectangular pulse from 0 to T
T = 0.2
kernel_t = np.arange(0, T + dt, dt)   # causal: t >= 0
h = np.ones_like(kernel_t)
h /= h.sum()  # Normalize

# Powers to analyze
powers = [1, 2, 3, -1, 0.5]

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('font', family='serif', size=12)

# Perform causal convolution for each power
for i, n in enumerate(powers):
    # Handle domain issues manually
    func = np.where((t != 0) | (n > 0), np.power(t, n, where=(t != 0) | (n > 0)), 0)
    func = np.nan_to_num(func, nan=0.0, posinf=0.0, neginf=0.0)

    # Causal convolution
    conv_result = convolve(func, h, mode='full')[:len(t)]

    # Time axis shift to match the result length (due to causal kernel)
    t_shifted = t

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_shifted, func, label=f'$t^{{{n}}}$ (Original)', linestyle='--', linewidth=2, color='gray')
    ax.plot(t_shifted, conv_result, label='Causal Convolution', linewidth=2, color='blue')

    ax.set_title(f'Causal Convolution of $t^{{{n}}}$ with Rectangular Kernel $h(t \\geq 0)$', fontsize=14)
    ax.set_xlabel('t')
    ax.set_ylabel('Amplitude')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', linewidth=0.5)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=11)
    plt.tight_layout()
    plt.show()

