import numpy as np
import matplotlib.pyplot as plt

N=1000
RANGE=(0,100)
trend=lambda x: x**2+2*x+2
sezon=lambda x: 200*np.sin(300*np.pi*x)+150*np.sin(80*np.pi*x)
noise=+np.random.normal(size=N,scale=100)
ens=np.linspace(RANGE[0],RANGE[1],N);
distance=(RANGE[1]-RANGE[0])/N
signal=trend(ens)+sezon(ens)+noise;


axs,plts=plt.subplots(4);

plts[0].plot(ens,signal);
plts[1].plot(ens,trend(ens));
plts[2].plot(ens,sezon(ens));
plts[3].plot(ens,noise);


plt.legend(["signal",'trend','seson','noise'])

plt.show()
plt.savefig("1.pdf")
