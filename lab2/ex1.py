
import numpy as np
import matplotlib.pyplot as plt

semnal_sin=lambda x,f:np.sin(f+2*np.pi*x)
semnal_cos=lambda x,f:np.cos(f+2*np.pi*x)
axs, fig=plt.subplots(2)

ens=np.linspace(0,2,100);
fig[0].plot(ens,semnal_sin(ens,0))
fig[1].plot(ens,semnal_cos(ens,-np.pi/2))

plt.show()
