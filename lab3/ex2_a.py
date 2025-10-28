import numpy as np
import matplotlib.pyplot as plt

tau=np.pi*2
circle=lambda x: np.e**(-1j*tau*x)


f= lambda x: np.sin(tau*x*5)
ens=np.linspace(0,1,100)
axs, fig=plt.subplots(1,2);
fig[0].plot(ens,f(ens));
rez=circle(ens)*f(ens)
fig[1].plot(rez.real,rez.imag);
plt.show()
