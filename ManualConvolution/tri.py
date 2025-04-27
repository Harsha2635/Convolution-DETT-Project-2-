# Harshil Rathan - EE24BTECH11064
import numpy as np 
import matplotlib.pyplot as plt

# defining Rectangular kernel 
def rectangular_kernel(N, T):
    h = np.zeros(N)
    n = np.arange(-N//2, N//2)
    h[(n >= -T) & (n <= T)] = 1
    return h

# sin as input 
def trigSIN(a , N):
    n = np.arange(-N//2, N//2)
    return np.sin(a * n)

# cos as input 
def trigCOS(a , N):
    n = np.arange(-N//2, N//2)
    return np.cos(a * n)

# tan as input 
def trigTAN(a , N):
    n = np.arange(-N//2, N//2)
    return np.tan(a * n)

# manual convolution
def manual_convolution(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    for n in range(len(y)):
        for k in range(M):
            if 0 <= n-k < N:
                y[n] += h[k] * x[n-k]
    return y

# Parameter defining 
N = 30
T = 10
a = 0.2
n = np.arange(-N//2, N//2)

b = trigSIN(a , N)
c = trigCOS(a , N)
d = trigTAN(a , N)

h = rectangular_kernel(N, T)

y = manual_convolution(b , h)
z = manual_convolution(c , h)
w = manual_convolution(d , h)

# New x-axis for convolution results
n_conv = np.arange(-(N-1), (N-1)+1)

plt.figure(figsize=(16,16))

plt.subplot(4,1,1)
plt.plot(n_conv, y)
plt.title("Manual Convolution of sin")
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(n_conv, z)
plt.title("Manual Convolution of cos")
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(n_conv, w)
plt.title("Manual Convolution of tan")
plt.grid(True)

plt.subplot(4,1,4)
plt.plot(n, h)
plt.title("Rectangular Kernel")
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()
