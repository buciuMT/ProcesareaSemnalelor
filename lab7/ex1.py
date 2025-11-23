import numpy as np
import matplotlib.pyplot as plt

fnormal=[
    lambda x,y:np.sin(2*np.pi*x+3*np.pi*y),
    lambda x,y:np.sin(4*np.pi*x)+np.cos(6*np.pi*y),
]

zero_tresh=0.001

ffft=[
    lambda x,y:np.logical_and(abs(np.fmod(y,5))<zero_tresh,abs(x)<zero_tresh),
    lambda x,y:np.logical_and(abs(np.fmod(x,5))<zero_tresh,abs(y)<zero_tresh),
    lambda x,y:np.logical_and(abs(np.fmod(x,5))<zero_tresh , abs(np.fmod(y,5))<zero_tresh),
]

apply=lambda mat,f:f(mat.real,mat.imag) 

N=64
C=8
mat_tmp=np.linspace(0,C,N,endpoint=False)
mat=np.linspace(mat_tmp,mat_tmp+C*1j,N,endpoint=False);

def runall(fn,arr,fmt):
    for i,f in enumerate(arr):
        fig,axs=plt.subplots(2)
        p=apply(mat,f)
        axs[0].imshow(p)
        ip=np.abs(fn(p))
        #ip=20*np.log10(ip)
        axs[1].imshow(ip)
        plt.savefig(fmt.format(i))
        plt.show()

runall(np.fft.fft2,fnormal,"normal_{}.pdf")
runall(np.fft.ifft2,ffft,"inverse_{}.pdf")

