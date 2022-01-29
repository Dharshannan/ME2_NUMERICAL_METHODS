import numpy as np
import matplotlib.pyplot as plt
# could also use for loops and math.comb (binomial coeff) to get difference
'''def Newdiff(x,y,k,l):
    if k == 1:
        return((y[l+1] - y[l])/(x[l+1] - x[l]))
    else:
        return((Newdiff(x,y,k-1,l+1) - Newdiff(x,y,k-1,l))/(x[k+l] - x[l]))
x = [1,2,3,4,5]
y = np.log(x)
print(Newdiff(x,y,2,0))'''

def Newdiff(y,k,l):
    if k == 1:
        return(y[l+1] - y[l])
    else:
        return(Newdiff(y,k-1,l+1) - Newdiff(y,k-1,l))


def derivative(x,y,k):
    h = x[1] - x[0]
    diff = Newdiff(y,k,0)
    der = diff/(h**k)
    return(der)

x = np.linspace(0,np.pi,100)
xn = np.linspace(0,np.pi,10)
yn = np.sin(xn)
k = 5 # fifth derivative

def Lagrange(xt, yt):
    lag = 0
    for j in range(0,len(xt)):
        mul = 1
        for k in range(0,len(xt)):
            if k != j:
                mul *= (x-xt[k])/(xt[j]-xt[k])
        lag += yt[j]*mul
    return(lag)

yt = Lagrange(xn,yn)

derv = np.zeros(len(x))
for i in range(0, len(x)-k):
    derv[i] = derivative(x[i:],yt[i:],k)

    
plt.scatter(x,yt)
plt.scatter(x,derv)
plt.show()    


    