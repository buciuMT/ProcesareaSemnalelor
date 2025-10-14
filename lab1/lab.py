import numpy as np
import matplotlib.pyplot as plt

#1
#a)
X=lambda t: np.sin(520*t*np.pi+np.pi/3)
Y=lambda t: np.sin(280*t*np.pi-np.pi/3)
Z=lambda t: np.sin(120*t*np.pi+np.pi/3)
#b)
L=np.arange(0,0.03,0.0005)
print(L)
LX=np.linspace(0,0.03,6)
LY=np.linspace(0,0.03,6)
LZ=np.linspace(0,0.03,6)

fig,axs=plt.subplots(3)
fig.suptitle("X Y and Z")
axs[0].plot(LX,X(LX))
axs[0].stem(LX,X(LX))
axs[1].plot(LY,Y(LY))
axs[1].stem(LY,Y(LY))
axs[2].plot(LZ,Z(LZ))
axs[2].stem(LZ,Z(LZ))
fig.show()
plt.show()
#c)
L200=np.linspace(0,1,200) # 200hz
fig,axs=plt.subplots(3)
fig.suptitle("X Y and Z")
scale=1/90
rng=np.linspace(0,scale,100)
axs[0].plot(rng,X(rng))
ens=L200[(L200>=0)&(L200<scale)]
axs[0].stem(ens,X(ens))

scale=1/50
rng=np.linspace(0,scale,100)
axs[1].plot(rng,Y(rng))
ens=L200[(L200>=0)&(L200<scale)]
axs[1].stem(ens,Y(ens))

scale=1/20
rng=np.linspace(0,scale,100)
axs[2].plot(rng,Z(rng))
ens=L200[(L200>=0)&(L200<scale)]
axs[2].stem(ens,Z(ens))

fig.show()
plt.show()
plt.show()

#2

#a
fig,axs=plt.subplots()
fig.suptitle("Un semnal sinusoidal de frecvent, ˘a 400 Hz, care s˘a cont, in˘a 1600 dees, antioane")
z=200
e=np.linspace(0,1/z,1600//z)
p=np.linspace(0,1/z,1600)
f=lambda t: np.sin(400*2*np.pi*t)

axs.plot(p,f(p))
axs.stem(e,f(e))
fig.show()
plt.show()
#b

fig,axs=plt.subplots()
fig.suptitle("Un semnal sinusoidal de frecvent, ˘a 800 Hz, care s˘a dureze 3 secunde.")
z=800
e=np.linspace(0,3/z,1600*3//z)
p=np.linspace(0,3/z,1600)
f=lambda t: np.sin(800*2*np.pi*t)

axs.plot(p,f(p))
axs.stem(e,f(e))
fig.show()
plt.show()

#c

fig,axs=plt.subplots()
fig.suptitle("Un semnal de tip sawtooth de frecvent, ˘a 240 Hz")
z=100
e=np.linspace(0,1/z,240*8//z)
p=np.linspace(0,1/z,1600)
f=lambda t: np.mod(t*240,1)
axs.plot(p,f(p))
axs.stem(e,f(e))
fig.show()
plt.show()


#d
fig,axs=plt.subplots()
fig.suptitle("Un semnal de tip square de frecvent, ˘a 300 Hz")
z=100
e=np.linspace(0,1/z,200*8//z)
p=np.linspace(0,1/z,1600)
f=lambda t: np.sign(np.sin(300*2*np.pi*t))
axs.plot(p,f(p))
axs.stem(e,f(e))
fig.show()
plt.show()

#e

fig,axs=plt.subplots()
fig.suptitle("Un semnal 2D aleator")
im=np.random.rand(128,128)
plt.imshow(im)
plt.show()

#f

def init():
    im=np.arange(128*128).reshape(128,128)
    imx=im%128
    imy=im//128
    im=np.sin(np.sqrt(np.pow(imx-64,2)+np.pow(imy-64,2)))
    return im
im=init()
fig,axs=plt.subplots()
fig.suptitle("Un semnal 2D propriu")
plt.imshow(im)
plt.show()
