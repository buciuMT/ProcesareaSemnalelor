import numpy as np
fs = 44100.
from scipy.io import wavfile as w

a=lambda t: np.sin(400*2*np.pi*t)
b=lambda t: np.sin(800*2*np.pi*t)
c=lambda t: np.mod(t*240,1)
d=lambda t: np.sign(np.sin(300*2*np.pi*t))

arr=[a,b,c,d]

ens=np.arange(0,2,1/fs)

for i,fnc in enumerate(arr):
    w.write(f"ex3_{i+1}.wav",int(fs),fnc(ens))

_,arr=w.read("ex3_1.wav");
print(f'arr==a(ens) {np.all(arr==a(ens))}')
