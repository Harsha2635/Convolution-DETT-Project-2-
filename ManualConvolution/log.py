import numpy as np
import matplotlib.pyplot as plt

# Define Rectangular Kernel
def rectangular_kernel(N, T):
    h = np.zeros(N)
    n = np.arange(-N//2, N//2)
    h[(n >= -T) & (n <= T)] = 1
    return h

# Define Logarithmic Function log(a*n + 1)
def log_func(a, N):
    n = np.arange(-N//2, N//2)
    result = np.zeros(N)
    valid_indices = n >= 0
    result[valid_indices] = np.log(a * n[valid_indices] + 1)
    return result

# Manual Convolution
def manual_convolution(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    for n in range(len(y)):
        for k in range(M):
            if 0 <= n-k < N:
                y[n] += h[k] * x[n-k]
    return y

# Parameters
N = 30
T = 10
a = 1

n = np.arange(-N//2, N//2)

x = log_func(a, N)
h = rectangular_kernel(N, T)

y = manual_convolution(x, h)

n_conv = np.arange(-(N-1), (N-1)+1)

# Plotting
plt.figure(figsize=(12,12))

plt.subplot(3,1,1)
plt.plot(n_conv, y)
plt.title("Manual Convolution of Logarithmic Function")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(n, x)
plt.title("Logarithmic Function (Input)")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(n, h)
plt.title("Rectangular Kernel")
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()
