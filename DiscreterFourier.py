import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [13,1]
from scipy.fftpack import rfft, rfftfreq, irfft
# Number of sample points
data_step   = 0.0005
x = np.arange(0,0.5,data_step)
y = np.sin(2*np.pi*50*x) + np.sin(2*np.pi*120*x)
plt.plot(x,y) # actual signal
# Introduce noise
noise = 2.5*np.random.random(len(y))
y = y + noise
# Perform transform
yf = rfft(y)
xf = rfftfreq(len(x), data_step)

plt.plot(x,y) # signal with noise

#plt.plot(xf, np.abs(yf)) # This gives the frequancy domain
plt.show()

# Cleaning the signal
yf_abs = np.abs(yf)
indices = yf_abs > 300 #(this value 300 is based of f domain plot)
yf_clean = indices * yf
#plt.plot(xf, np.abs(yf_clean))

clean = irfft(yf_clean)
plt.plot(x,clean)

