#Harshil Rathan - EE24BTECH11064
import numpy as np 
import matplotlib.pyplot as plt

#defining Rectangular kernel 
def rectangular_kernel(N, T):
    h = np.zeros(N)
    n = np.arange(-N//2, N//2)
    h[(n >= -T) & (n <= T)] = 1
    return h

#tan-1 as input 
def inversetrigTAN(a , N):
    n = np.arange(-N//2, N//2)
    return np.arctan(a * n)

#sin-1 as input 
def inversetrigSIN(a , N):
    n = np.arange(-N//2, N//2)
    x = np.clip(a * n , -1, 1)
    return np.arcsin(x)

#cos-1 as input 
def inversetrigCOS(a , N):
    n = np.arange(-N//2, N//2)
    x = np.clip(a * n , -1, 1)
    return np.arccos(x)

def CircularConvo(x , h):
#fast fourier transform and inverse 
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(h)))  

#Parameter defining 
N = 30 
T = 10 
a = 0.03
n = np.arange(-N//2, N//2)

b = inversetrigTAN(a , N)
c = inversetrigSIN(a , N)
d = inversetrigCOS(a , N)

h = rectangular_kernel(N, T)

y = CircularConvo(b , h)
z = CircularConvo(c , h)
w = CircularConvo(d , h)

plt.figure(figsize=(16,16))

plt.subplot(4,1,1)
plt.plot(n,y)
plt.title("CC of TAN-1")
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(n,z)
plt.title("CC of SIN-1")
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(n,w)
plt.title("CC of COS-1")
plt.grid(True)

plt.subplot(4,1,4)
plt.plot(n,h)
plt.title("Rectangular Kernel")
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()
