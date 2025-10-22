from numpy import pi
import numpy as np
from scipy.io import wavfile as w

f=lambda x,f:np.cos(f*2*np.pi*x)
freq=44100
ens=np.linspace(0,3,3*freq)
w.write("ex5.wav",freq,(f(ens,700)+f(ens,800))/2) #RoAlert

w.write("ex5_2.wav",freq,np.concatenate((f(ens,400),f(ens,500))))
# doua note in succesiune
