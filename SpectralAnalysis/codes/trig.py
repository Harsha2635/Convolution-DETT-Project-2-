import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

# Define time and rectangular pulse parameters
T = 1.0  # Width of h(t)
t = np.linspace(-10, 10, 5000)  # Time vector
h = np.where(np.abs(t) <= T, 1.0, 0.0)  # Rectangular pulse

# Define sinusoidal signals with increasing frequencies
frequencies = [0.2, 0.3,0.5]  # in Hz
sinusoids = [np.sin(2 * np.pi * f * t) for f in frequencies]

# === 1. Plot the 3 original sinusoids ===
plt.figure(figsize=(12, 6))
for i, f in enumerate(frequencies):
    plt.plot(t, sinusoids[i], label=f'{f} Hz')
plt.title('Original Sinusoidal Signals')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.axhline(0, color='black', linewidth=0.8)  # y-axis
plt.axvline(0, color='black', linewidth=0.8)  # x-axis
plt.legend()
plt.grid(True)
plt.savefig('../figs/sin1.png')
plt.show()

# === 2. Convolve with h(t) and plot the convolution results ===
convolved_signals = []
plt.figure(figsize=(12, 6))
for i, signal in enumerate(sinusoids):
    conv_signal = np.convolve(signal, h, mode='same') * (t[1] - t[0])  # Convolve and scale
    convolved_signals.append(conv_signal)
    plt.plot(t, conv_signal, label=f'Convolved {frequencies[i]} Hz')
plt.title('Convolution with Rectangular Pulse h(t)')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.axhline(0, color='black', linewidth=0.8)  # y-axis
plt.axvline(0, color='black', linewidth=0.8)  # x-axis
plt.legend()
plt.grid(True)
plt.savefig('../figs/sin2.png')
plt.show()

# === 3. FFT of the Convolved Signals ===
plt.figure(figsize=(12, 6))
for i, conv_signal in enumerate(convolved_signals):
    Y = fftshift(fft(conv_signal))
    freq = fftshift(fftfreq(len(t), d=(t[1]-t[0])))
    plt.plot(freq, np.abs(Y), label=f'Convolved {frequencies[i]} Hz')
plt.title('Spectral Magnitude of Convolved Signals')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim([-10, 10])  # Focus view on low frequencies
plt.axhline(0, color='black', linewidth=0.8)  # y-axis
plt.axvline(0, color='black', linewidth=0.8)  # x-axis
plt.legend()
plt.savefig('../figs/sin3.png')
plt.grid(True)
plt.show()

