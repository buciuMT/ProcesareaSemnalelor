import numpy as np
import matplotlib.pyplot as plt


f=lambda x,f:np.sin(f*2*np.pi*x)
ens_freq=10000
ens=np.linspace(0,1,ens_freq)
ens=np.linspace(0,1,ens_freq)

gr=np.linspace(0,1,100*ens_freq)
decgra=gr[::4]
decgra=decgra[decgra<=4.1/ens_freq]

decgrb=gr[1::4]
decgrb=decgrb[decgrb<=4.1/ens_freq]
gr=gr[gr<=4.1/ens_freq]
ensa=ens[ens<=4.1/ens_freq]
dec_ens=ens[::4]
dec_ensb=ens[1::4]
dec_ensa=dec_ens[dec_ens<=4.1/ens_freq]
dec_ensb=dec_ensb[dec_ensb<=4.1/ens_freq]

axs, fig=plt.subplots(3)

fig[0].plot(gr,f(gr,ens_freq))
fig[0].stem(ensa,f(ensa,ens_freq))

fig[1].plot(decgra,f(decgra,ens_freq))
fig[1].stem(dec_ensa,f(dec_ensa,ens_freq))

fig[2].plot(decgrb,f(decgrb,ens_freq))
fig[2].stem(dec_ensb,f(dec_ensb,ens_freq))



plt.show()
