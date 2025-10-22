import matplotlib.pyplot as plt
import numpy as np
c=lambda t: np.mod(t*240,1)
d=lambda t: np.sin(300*2*np.pi*t)
ens=np.arange(0,1,1/80000);
ens=ens[ens<=1/100]

print(ens)
plt.plot(ens,c(ens)+d(ens))
plt.show()
