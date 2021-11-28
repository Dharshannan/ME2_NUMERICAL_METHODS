#Cubic splines
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1,100)
#y = np.multiply(x,np.exp(-x))
y = 1/(1+25*x**2)
h = 0.125
xt = np.arange(-1,1+h,h)
#yt = np.multiply(xt,np.exp(-xt))
yt = 1/(1+25*xt**2)
list1 = [i for i in range(0,len(xt)+1)]
n = len(xt)
a = np.zeros(n-1)
b = np.zeros(n-1)
c = np.zeros(n-1)
d = np.zeros(n-1)

# v coeff values for free end condition

N = n-1
M = np.zeros((n,n))
M[0][0] = 2
M[0][1] = 1
M[N][N-1] = 1
M[N][N] = 2
#print(M)

for i in range(1,len(M)-1):
    M[i][list1[i-1]] = 1/(xt[i] - xt[i-1])
    M[i][list1[i]] = 2*((1/(xt[i]-xt[i-1]) + 1/(xt[i+1]-xt[i])))
    M[i][list1[i+1]] = 1/(xt[i+1] - xt[i])
    
#print(M)

# Mv = B, Matrix B:

B = np.zeros((n,1))
B[0] = (3*(yt[1]-yt[0]))/(xt[1]-xt[0])
B[N] = (3*(yt[N]-yt[N-1]))/(xt[N]-xt[N-1])

for i in range(1,n-1):
    B[i] = 3*(((yt[i]-yt[i-1])/(xt[i]-xt[i-1])**2) + ((yt[i+1]-yt[i])/(xt[i+1]-xt[i])**2))
    
#print(B)

#Gauss elim to get v values or just use numpy solver lol

v = np.linalg.solve(M,B)
#print(v)

for i in range(0,n-1):
    a[i] = yt[i]
    b[i] = v[i]
    c[i] = 3*((yt[i+1]-yt[i])/(xt[i+1]-xt[i])**2) - ((v[i+1]+2*v[i])/(xt[i+1]-xt[i]))
    d[i] = -2*((yt[i+1]-yt[i])/(xt[i+1]-xt[i])**3) + ((v[i+1]+v[i])/(xt[i+1]-xt[i])**2)
    
    #cubic splines functions
    xx = np.linspace(xt[i],xt[i]+h,100)
    q = a[i] + b[i]*(xx-xt[i]) + c[i]*(xx-xt[i])**2 + d[i]*(xx-xt[i])**3
    plt.plot(xx,q)
    
plt.plot(x,y, c='b')    

