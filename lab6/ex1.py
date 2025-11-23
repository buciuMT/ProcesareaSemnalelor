import numpy as np
import matplotlib.pyplot as plt
gen_sinc= lambda b:lambda t: np.sinc(b*t)**2

R=(-3,3)

hf=[1,1.5,2,4]

x=gen_sinc(1)

tens=np.linspace(R[0],R[1],100)
tfnc=x(tens)

axs,fig=plt.subplots(len(hf))

for i,freq in enumerate(hf):
    ens=np.arange(1/freq*np.floor(R[0]*freq),R[1],1/freq)
    ts=1/freq
    img=x(ens)
    fig[i].plot(tens,tfnc,'g--')
    fig[i].stem(ens,img)
    xr=lambda t: np.sum([np.sinc(t-e) for e,v in zip(ens,img)],axis=1)
    fig[i].plot(tens,xr(tens))
    


plt.show()
