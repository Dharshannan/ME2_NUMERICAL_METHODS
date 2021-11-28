import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1,5,100)
y = np.log(x)

x_t = [1, 2, 3, 5]
y_t = np.log(x_t)


def Newton_diff(xt, yt, k, l):
    if k - l == 1:
        return((yt[k] - yt[l])/(xt[k] - xt[l]))
    else:
        return((Newton_diff(xt, yt, k, l+1)) - Newton_diff(xt, yt, k-1, l))/(xt[k] - xt[l])


print(Newton_diff(x_t, y_t, 3, 0))

def interpolate(x_t, y_t):
    poly = y_t[0]
    mul = (x-x_t[0])
    for i in range(1, len(x_t)):
        poly += Newton_diff(x_t, y_t, i, 0)*mul
        mul *= (x-x_t[i])
    return(poly)
    
plt.plot(x,y)
plt.plot(x,interpolate(x_t, y_t))
plt.show()

