import numpy as np
from typing import Callable
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
N=1000
RANGE=(0,100)
trend:Callable[[np.ndarray],np.ndarray]=lambda x: x**2+2*x+2
sezon:Callable[[np.ndarray],np.ndarray]=lambda x: 200*np.sin(300*np.pi*x)+150*np.sin(80*np.pi*x)
noise=+np.random.normal(size=N,scale=100)
ens=np.linspace(RANGE[0],RANGE[1],N);
distance=(RANGE[1]-RANGE[0])/N
signal=trend(ens)+sezon(ens)+noise;

p=3
ar=AutoReg(signal,p)
print(ar.fit().summary())


