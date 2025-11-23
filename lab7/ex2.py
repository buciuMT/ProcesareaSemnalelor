import scipy.datasets as misc
import numpy as np
import matplotlib.pyplot as plt

racoon=misc.face(gray=True)
plt.imshow(racoon);
plt.show()
imff=np.fft.fft2(racoon)
imff[abs(imff)<1000000]=0.
plt.imshow(np.abs(np.fft.ifft2(imff)))
plt.show()
