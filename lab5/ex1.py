import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt("Train.csv",delimiter=',')
x=data[1:,2]
id=data[1:,0]


axs,fig=plt.subplots(4);


#a) Care este frecvent, a de es, antionare a semnalului din Train.csv
# frecventa este de un ensantion pe ora sau 1/3600 hz
f_sample=1/3600

#b) intervalul de timp acoperit de ensantioane este (1/f_sample)*len(x)=65840400 secunde sau len(x)=18289
#pentru forier ultimul element nu este ensantionat deoarece se repeta
print(f'{(1/f_sample)*len(x)} secunde')
print(f"{len(x)} ore")
ens=np.linspace(0,(1//f_sample)*len(x),len(x));
freqs=f_sample*np.linspace(0,len(x)/2,len(x)//2)/len(x)

#c) frecventa maxima conform Nyquist este f_sample/2=1/7200

#d)
X=np.fft.fft(x)
X_m=X[:len(x)//2]
fig[0].scatter(freqs,np.abs(X_m))
fig[0].stem(freqs,np.abs(X_m))


#e) Da eliminam X[0]
newX=X
newX[0]=0
fig[1].plot(np.fft.ifft(newX))



#f) 

mxindx=np.argpartition(np.abs(X_m),-4)[-4:]
print(freqs[mxindx])
fig[2].scatter(freqs,np.abs(X_m))
fig[2].stem(freqs,np.abs(X_m))
fig[2].stem(freqs[mxindx],np.abs(X_m)[mxindx])

mxfreqs=freqs[mxindx]
print(f'hz {mxfreqs}')
print(f'ore {1/(mxfreqs)/3600}')
print(f'zile {1/(mxfreqs)/3600/24}')
print(f'saptamani {1/(mxfreqs)/3600/24/7}')
print(f'luni~ {1/(mxfreqs)/3600/24/30.44}')
print(f'ani {1/(mxfreqs)/3600/24/365}')

#f) se pare ca corespund  cu o zi un an, doi ani si undeva pe la 8 luni? vacanta

#g)
LUNI_IDX=1056
ZI=24
LUNA=ZI*30
nx=x[LUNI_IDX:(LUNI_IDX+LUNA)]
print(len(nx))
fig[3].plot(nx)
plt.savefig("ex1g.pdf")
plt.show()
#h)
'''
Putem sa calculam data exacta folosindu-ne de patternurile din weekend pentru a delimita saptamaniile, ratele de utilizare specifice in zilele de sarbatoare, cat si statistic numarul de masini care ar putea trece pentru a afla anul.
Pentru calcului in cadrul unui oras putem sa calculam anul aproximativ folosindu-ne de numarul de masini din oras
'''
