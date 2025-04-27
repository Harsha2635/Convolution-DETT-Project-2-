import numpy as np
import matplotlib.pyplot as plt

# Define Rectangular Kernel
def rectangular_kernel(N, T):
    h = np.zeros(N)
    n = np.arange(-N//2, N//2)
    h[(n >= -T) & (n <= T)] = 1
    return h

# Define Algebraic Function (a * n)^k
def algebraic_func(a, N, k):
    n = np.arange(-N//2, N//2)
    return (a * n) ** k

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
k = 1

n = np.arange(-N//2, N//2)

x = algebraic_func(a, N, k)
h = rectangular_kernel(N, T)

y = manual_convolution(x, h)

n_conv = np.arange(-(N-1), (N-1)+1)

# Plotting
plt.figure(figsize=(12,12))

plt.subplot(3,1,1)
plt.plot(n_conv, y)
plt.title("Manual Convolution of Algebraic Function")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(n, x)
plt.title("Algebraic Function (Input)")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(n, h)
plt.title("Rectangular Kernel")
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()
