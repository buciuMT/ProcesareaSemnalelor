import matplotlib.pyplot as plt
from numpy import arange, linspace, pi
import numpy as np

f=lambda x,f:np.cos(f*2*np.pi*x)
ens_freq=1
ens=linspace(0,7.01,200)
gr=arange(0,7.01,1/ens_freq)


axs, fig=plt.subplots(3)
fig[0].plot(ens,f(ens,ens_freq))
fig[0].stem(gr,f(gr,ens_freq))
fig[1].plot(ens,f(ens,ens_freq/2))
fig[1].stem(gr,f(gr,ens_freq/2))
fig[2].plot(ens,f(ens,0))
fig[2].plot(gr,f(gr,0))

plt.show()

#viteza de ensantoniare conteaza
