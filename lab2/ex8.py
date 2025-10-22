import numpy as np
import matplotlib.pyplot as plt

aprox=lambda x: (x-(7*(x**3)/60))/(1+(x**2)/20)
ens=np.linspace(-np.pi,np.pi,400)
t=np.sin(ens)
f=aprox(ens)
d=np.abs(t-f)


axs, fig=plt.subplots(3)

fig[0].plot(ens,t)
fig[1].plot(ens,f)
fig[2].plot(ens,d)

fig[2].set_yscale("log")

plt.show()
