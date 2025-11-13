import fft
import numpy as np
import matplotlib.pyplot as plt
import time


freqs=[(2,4,0),(3,.5,0),(7,1,0),(8,2,0),(13,8,0)]
gen=lambda freqs: lambda x: np.sum(np.fromiter((a*np.sin(2*np.pi+f*x+o) for (f,a,o) in freqs),np.dtype))

f=gen(freqs)

counts=[128, 256, 512, 1024, 2048, 4096, 8192]

np_fft=[]
mat_ft=[]
my_fft=[]

for elem in counts:
    ens=np.linspace(0,1,elem)
    print(f"iter{elem}")
    vec=f(ens)
    start=time.time()
    _ =np.fft.fft(vec)
    stop=time.time()
    np_fft.append(stop-start)

for elem in counts:
    ens=np.linspace(0,1,elem)
    print(f"iter{elem}")
    vec=f(ens)
    mat=fft.gen_mat(elem)
    start=time.time()
    _ =vec@mat
    stop=time.time()
    mat_ft.append(stop-start)

for elem in counts:
    ens=np.linspace(complex(0),complex(1),elem)
    print(f"iter{elem}")
    vec=f(ens)
    true =np.fft.fft(vec)
    start=time.time()
    verf=fft.fft(vec)
    stop=time.time()
    if not np.allclose(true,verf):
        print("error")
        exit(-1)
    my_fft.append(stop-start)


plts=[
    plt.plot(np_fft),
    plt.plot(mat_ft),
    plt.plot(my_fft)]

plt.yscale('log')

plt.legend(['numpy','matrix','custom fft'])

plt.savefig("ex1.pdf")
plt.show()
