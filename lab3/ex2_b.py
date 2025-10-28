import numpy as np
import matplotlib.pyplot as plt

tau=np.pi*2
circle=lambda x: np.e**(-1j*tau*x)

axs, fig=plt.subplots(2,2);

for (idx,freq) in enumerate([1,2,5,7]):
    f= lambda x: np.sin(tau*x*3)
    ens=np.linspace(0,1,500)
    rez=circle(ens*freq)*f(ens)
    fig[idx//2][idx%2].plot(rez.real,rez.imag);

plt.show()
