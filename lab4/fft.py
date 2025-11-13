import numpy as np


gen_mat=lambda N:np.matrix([[(np.e**(-2*np.pi*1j*i*j/N))/np.sqrt(N) for i in range(N)] for j in range(N)])




def fft(v:np.ndarray,inv=False):
    l=len(v)
    if l==1:
        return v
    even=fft(v[::2],inv)
    odd=fft(v[1::2],inv) 
    w=np.exp(-2j*np.pi*np.arange(0,l//2)/l*(-1 if inv else 1))
    term=w*odd
    return np.concat([even+term,even-term])
