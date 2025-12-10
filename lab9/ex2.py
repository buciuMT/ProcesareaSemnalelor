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

np.random.seed(1337)
noise2=np.random.normal(N,scale=10)


alpha=1


def generate_s(alpha:float,signal:np.ndarray):
    s=np.zeros_like(signal)
    s[0]=signal[0]
    for i in range(1,len(s)):
        s[i]=alpha*signal[i]+(1-alpha)*s[i-1]
    return s
s=generate_s(alpha,signal)

class MAmodel:
    def __init__(self,n:int):
        self.n=n
        self.alphas=np.zeros(n)
    def predict(self,data:np.ndarray,indx:int=0):
        res=np.zeros_like(data);
        if indx>=self.n:
            return res
        for offset,i in enumerate(range(indx,self.n)):
            s=generate_s(self.alphas[i],data)
            res[offset+1:]+=s
            data:np.ndarray=data[:-1]-data[1]
        return res


    def train(self,data:np.ndarray,indx:int=0):
        if indx>=self.n:
            return 
        data2=data[:-1]-data[1:]
        self.train(data2,indx+1)
        possible_values=np.linspace(0,1,1000,True)
        best_value=(np.inf,0)
        for alpha in possible_values:
            self.alphas[indx]=alpha




plt.plot(s)
plt.plot(signal)
plt.show()




#DEZASTRU TREBUIE REPARAT
