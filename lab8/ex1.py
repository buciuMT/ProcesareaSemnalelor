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

plt.plot(ens,signal);
plt.plot(ens,trend(ens));
plt.plot(ens,sezon(ens));
plt.plot(ens,noise);


plt.legend(["signal",'trend','seson','noise'])

plt.show()
plt.savefig("a.pdf")


true_corelation=np.correlate(signal,signal,mode='full');

my_corelation=np.convolve(np.pad(signal,(N,),'constant',constant_values=0),np.flip(signal),mode='valid') #numar real nu necesita conjugare complexa

plt.plot(true_corelation)
plt.plot(my_corelation)

plt.legend(["np.correlate",'corelation with convolve'])

plt.show()
plt.savefig("b.pdf")

def find_x(signal:np.ndarray,p:int,m:int):
    #m - orizontul de timp
    #y - signal
    #y=Y*x
    #x=Yt*y
    y=signal[-m:]#np.reshape(signal[-m:],(-1,1))
    Y=np.array([[signal[-1-offset-parm] for parm in range(p)] for offset in range(m)])
    #np.linalg.inv gresit
    #least square
    #x=np.linalg.inv(Y.T@Y)@Y.T@y
    x=np.linalg.lstsq(Y,y)[0]
    return x

def predict_next(signal,x):
    return np.convolve(signal,x,mode='valid');

x=find_x(signal,2,20)
print(*x,sep='\n')
predicted=predict_next(signal,x)
plt.plot(ens[1:]-distance,predicted);
plt.plot(ens,signal);
plt.savefig("c.pdf")

plt.show()


best=(999999999999999,2,100)

for p in range(2,20):
    for m in range(100,900,10):
        x=find_x(signal,p,m);
        predicted=predict_next(signal[:-1],x)
        plt.plot(signal[1:])
        plt.plot(predicted)
        plt.show()
        err=sum((signal[p:]-predicted)**2)/N
        if err<best[0]:
            best=(err,p,m)

print(best) #(np.float64(17371.125919502632), 3, 110)




