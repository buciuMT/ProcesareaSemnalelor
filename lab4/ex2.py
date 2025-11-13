import numpy as np
import matplotlib.pyplot as plt

F=5
genf=lambda F:lambda x:np.sin(2*np.pi*x*F)

tens=np.linspace(0,1,500)
ens=np.arange(0,1,1/(2*F+1))
#Nyquist necesita ca f_sampling >= componenta de frecventa maxima, insa exista problema de fase lock
axs,fig=plt.subplots(3)

f=genf(F)
fig[0].plot(tens,f(tens))
fig[0].stem(ens,f(ens))

f=genf(3*F+1)
fig[1].plot(tens,f(tens))
fig[1].stem(ens,f(ens))

f=genf(5*F+2)
fig[2].plot(tens,f(tens))
fig[2].stem(ens,f(ens))

plt.savefig("ex2.pdf")
plt.show()
