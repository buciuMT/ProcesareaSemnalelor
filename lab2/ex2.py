import numpy as np
import matplotlib.pyplot as plt

A=1
F=2
O=0

semnal=lambda x,f:A*np.cos(f+F+2*np.pi*x)+O
axs, fig=plt.subplots(4)
ens=np.linspace(0,2,100);
fig[0].plot(ens,semnal(ens,1))
fig[1].plot(ens,semnal(ens,0.2))
fig[2].plot(ens,semnal(ens,0.5))
fig[3].plot(ens,semnal(ens,0.6))


plt.show()

norm_signal=(semnal(ens,0)**2).sum()
noise=np.random.random(ens.size)
norm_noise=(noise**2).sum()

axs, fig=plt.subplots(4)
for i,nsr in enumerate([0.1, 1, 10, 100]):
    ratio=norm_signal/nsr
    amplification=np.sqrt(nsr/ratio)
    fig[i].plot(ens,semnal(ens,1)+noise*amplification)

plt.show()

