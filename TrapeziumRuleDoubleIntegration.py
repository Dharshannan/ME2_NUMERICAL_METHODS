import numpy as np
import matplotlib.pyplot as plt

def trapseqd(xt,yt):
    n = len(xt)
    b = xt[n-1]
    a = xt[0]
    mid_sum = 0
    for i in range(1, n-1):
        mid_sum += yt[i]  
    trp = ((b-a)/(n-1))*(yt[0]*0.5 + yt[n-1]*0.5 + mid_sum)
    return(trp)

x = np.arange(-5+0.05,5,0.05)
G = np.zeros(len(x))

for i in range(0, len(x)):
    p = np.sqrt(25-x[i]**2)
    y = np.arange(-p+0.05,p,0.05)
    z = np.zeros(len(y))
    for j in range(0,len(y)):
        #z[j] = np.sqrt(5 - np.sqrt(x[i]**2 + y[j]**2))
        z[j] = np.sqrt(25 - (x[i]**2 + y[j]**2))
        
    G[i] = trapseqd(y,z)
    
I = trapseqd(x,G)
print(I)


