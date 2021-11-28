import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1,100)
y = 1/(1+25*x**2)
h = 0.125
x_t = np.arange(-1,1+h,h)
y_t = 1/(1+25*x_t**2)

def Lagrange(xt, yt):
    lag = 0
    for j in range(0,len(xt)):
        mul = 1
        for k in range(0,len(xt)):
            if k != j:
                mul *= (x-xt[k])/(xt[j]-xt[k])
        lag += yt[j]*mul
    return(lag)

lag_int = Lagrange(x_t, y_t)

plt.plot(x, lag_int, c='r')
plt.plot(x, y, c='b')
plt.xlim([-1,1])
plt.ylim([-1,1.5])
plt.show()

