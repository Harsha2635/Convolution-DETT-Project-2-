#Harshil Rathan - EE24BTECH11064
import numpy as np 
import matplotlib.pyplot as plt

#defining Rectangular kernel 
def rectangular_kernel(N, T):
    h = np.zeros(N)
    n = np.arange(-N//2, N//2)
    h[(n >= -T) & (n <= T)] = 1
    return h

#sin as input 
def trigSIN(a , N):
    n = np.arange(-N//2, N//2)
    return np.sin(a * n)

#cos as input 
def trigCOS(a , N):
    n = np.arange(-N//2, N//2)
    return np.cos(a * n)

#tan as input 
def trigTAN(a , N):
    n = np.arange(-N//2, N//2)
    return np.tan(a * n)

#cc
def CircularConvo(x , h):
#fast fourier transform and inverse 
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(h)))  

#Parameter definign 
N = 30
T = 10
a = 0.2
n = np.arange(-N//2, N//2)

b = trigSIN(a , N)
c = trigCOS(a , N)
d = trigTAN(a , N)

h = rectangular_kernel(N, T)

y = CircularConvo(b , h)
z = CircularConvo(c , h)
w = CircularConvo(d , h)

plt.figure(figsize=(16,16))

plt.subplot(4,1,1)
plt.plot(n,y)
plt.title("CC of SIN")
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(n,z)
plt.title("CC of COS")
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(n,w)
plt.title("CC of TAN")
plt.grid(True)

plt.subplot(4,1,4)
plt.plot(n,h)
plt.title("Rectangular Kernel")
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()