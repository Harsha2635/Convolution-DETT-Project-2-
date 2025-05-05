import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
t = np.linspace(-2, 2, 1000)
dt = t[1] - t[0]

# Define kernel h(t) as a rectangular pulse from -T to T
T = 0.2
kernel_t = np.arange(-T, T + dt, dt)
h = np.ones_like(kernel_t)
h /= h.sum()  # Normalize

# Powers to analyze
powers = [1, 2, 3, -1, 0.5]

# Standard plotting style
plt.style.use('seaborn-v0_8-whitegrid')
font = {'family': 'serif', 'size': 12}
plt.rc('font', **font)

# Iterate over each power
for i, n in enumerate(powers):
    # Handle domain errors (e.g., t^-1, t^0.5 for negative t)
    func = np.where((t != 0) | (n > 0), np.power(t, n, where=(t != 0) | (n > 0)), 0)
    func = np.nan_to_num(func, nan=0.0, posinf=0.0, neginf=0.0)

    # Convolution
    conv_result = convolve(func, h, mode='same')

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, func, label=f'$t^{{{n}}}$', linestyle='--', linewidth=2, color='tab:gray')
    ax.plot(t, conv_result, label='Convolved', linewidth=2, color='tab:blue')

    ax.set_title(f'Convolution of $t^{n}$ with Rectangular Kernel $h(t)$', fontsize=14)
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=11)

    plt.tight_layout()
    filename = f'../figs/t_pow_{str(n).replace(".", "_").replace("-", "neg")}.png' 
    plt.show()

