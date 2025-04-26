import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

# Define parameters
T = 1.0
t = np.linspace(-10, 10, 5000)
h = np.where(np.abs(t) <= T, 1.0, 0.0)

# Different functions
sinusoidal = np.sin(2 * np.pi * 1.0 * t)
algebraic = t**2
exponential = np.exp(-t) * (t >= 0)
t_shifted = t + 2
logarithmic = np.zeros_like(t)
valid_idx = t_shifted > 0
logarithmic[valid_idx] = np.log(t_shifted[valid_idx])
inverse_trig = np.arctan(t)

signals = {
    "Sinusoidal": sinusoidal,
    "Algebraic": algebraic,
    "Exponential": exponential,
    "Logarithmic": logarithmic,
    "Inverse Trig": inverse_trig
}

# Convolution and FFT
fig_counter = 0

for name, f in signals.items():
    y = np.convolve(f, h, mode='same') * (t[1] - t[0])
    
    # FFT
    Y = fftshift(fft(y))
    freq = fftshift(fftfreq(len(t), d=(t[1]-t[0])))

    # Time domain plot
    plt.figure(figsize=(7, 5))
    plt.plot(t, f, label='Original Signal', linestyle='--')
    plt.plot(t, y, label='Convolved Signal', linewidth=2)
    plt.title(f'Time Domain: {name}')
    plt.xlabel('Time (t)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../figs/Analysis{fig_counter}.png')
    plt.close()
    fig_counter += 1

    # Frequency domain plot
    plt.figure(figsize=(7, 5))
    plt.plot(freq, np.abs(Y))
    plt.title(f'Spectral Magnitude: {name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim([-10, 10])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../figs/Analysis{fig_counter}.png')
    plt.close()
    fig_counter += 1
