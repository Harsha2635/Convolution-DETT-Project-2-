#Harshil Rathan - EE24BTECH11064
import numpy as np 
import matplotlib.pyplot as plt

#defining Rectangular kernel 
def rectangular_kernel(N, T):
    h = np.zeros(N)
    n = np.arange(-N//2, N//2)
    h[(n >= -T) & (n <= T)] = 1
    return h

#exponential input 
def exponent(a , N):
    n = np.arange(-N//2, N//2)
    return np.exp(a * n)

def CircularConvo(x , h):
#fast fourier transform and inverse 
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(h)))  

#parameters defining 
N = 30 
T = 10 
a = 0.2

n = np.arange(-N//2, N//2)

x = exponent(a , N)
h = rectangular_kernel(N, T)

y = CircularConvo(x , h)

plt.figure(figsize=(12,12))

plt.subplot(3,1,1)
plt.plot(n,y)
plt.title("CC of Exponential Function")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(n,x)
plt.title("Exponential Func")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(n,h)
plt.title("Rectangular Kernel")
plt.grid(True)


plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()