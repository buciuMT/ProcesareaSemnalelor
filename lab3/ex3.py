import numpy as np
import matplotlib.pyplot as plt


N=100
#(F,A,O)
freqs=[(2,4,0),(3,.5,0),(7,1,0),(8,2,0),(13,8,0)]

gen=lambda freqs: lambda x: np.sum(np.fromiter((a*np.sin(2*np.pi+f*x+o) for (f,a,o) in freqs),np.dtype))
f=gen(freqs)

mat=np.matrix([[(np.e**(-2*np.pi*1j*i*j/N))/np.sqrt(N) for i in range(N)] for j in range(N)])

axs,fig = plt.subplots(2,2);

true=np.linspace(0,1,100)
fig[0][0].plot(true,f(true))
fig[0][0].scatter(np.linspace(0,1,N),f(np.linspace(0,1,N)))

ens=np.linspace(0,1,N);
res=f(ens)
ft=np.abs(mat@res).A1
fig[0][1].stem(ens,ft)

#Verificare
ft2=np.abs(np.fft.fft(res))
print(np.allclose(np.abs(ft),ft2/np.sqrt(N)))
fig[1][0].stem(ens,ft2)

rfreq=[(f,a,0) for f,a in zip(ens ,ft)]
g=gen(rfreq)
fig[1][1].plot(ens,g(ens))


plt.show()
