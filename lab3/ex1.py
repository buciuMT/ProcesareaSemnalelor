import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline 
N=8
f_1=5
f_2=20
f_3=75
t_s=1

mat=np.matrix([[(np.e**(-2*np.pi*1j*i*j/N))/np.sqrt(N) for i in range(N)] for j in range(N)])

axs,fig=plt.subplots(N)

ens=np.linspace(0,N-1,N*50)

for i in range(N):
    line=mat[:,i].A1
    s=make_interp_spline(range(N),line,k=2)
    fig[i].scatter(range(N),line.real)
    fig[i].plot(ens,s(ens).real)
    fig[i].scatter(range(N),line.imag)
    fig[i].plot(ens,s(ens).imag)

plt.show()

print(np.allclose(mat@mat.H,np.eye(N,N)))
