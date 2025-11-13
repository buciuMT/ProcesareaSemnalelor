import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

SR=44100 #hz

#a
_,arr=wav.read("vocale.wav")

N=len(arr)
print(N)

gs=N//1000


a_mat=[]

start=0
while start+gs<=N:
    a_mat.append(np.abs(np.fft.fft(arr[start:start+gs])))
    start=start+gs

mat=np.column_stack(a_mat)

plt.imshow(mat)

plt.savefig(f"ex6_{gs}.pdf",)

plt.show()

