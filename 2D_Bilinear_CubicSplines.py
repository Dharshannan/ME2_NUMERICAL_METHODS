import numpy as np
import matplotlib.pyplot as plt
np.seterr(over='ignore')

flower = plt.imread('Flower.jpg')
#plt.imshow(flower)
#plt.show()
M = np.copy(flower)
k = 2

flower_new = M[::k, ::k, ::] # Compress the image

def CubicSpline(xt,yt,xx):
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
        
    #print(M) #this gives the matrix of v coefficients
    
    # Mv = B, Matrix B:
    
    B = np.zeros((n,1))
    B[0] = (3*(yt[1]-yt[0]))/(xt[1]-xt[0])
    B[N] = (3*(yt[N]-yt[N-1]))/(xt[N]-xt[N-1])
    
    for i in range(1,n-1):
        B[i] = 3*(((yt[i]-yt[i-1])/(xt[i]-xt[i-1])**2) + ((yt[i+1]-yt[i])/(xt[i+1]-xt[i])**2))
        
    #print(B) #this gives the matrix values to evaluate v 
    
    #Gauss elim to get v values or just use numpy solver lol
    
    v = np.linalg.solve(M,B)
    #print(v)
    
    q = np.zeros(len(x))
    for i in range(0,n-1):
        a[i] = yt[i]
        b[i] = v[i]
        c[i] = 3*((yt[i+1]-yt[i])/(xt[i+1]-xt[i])**2) - ((v[i+1]+2*v[i])/(xt[i+1]-xt[i]))
        d[i] = -2*((yt[i+1]-yt[i])/(xt[i+1]-xt[i])**3) + ((v[i+1]+v[i])/(xt[i+1]-xt[i])**2)
        
        #cubic splines functions
        xxx = xx[(xt[i]<=x) & (x<=xt[i+1])] #taking x values between 2 nodes for each iteration
        q[(xt[i]<=x) & (x<=xt[i+1])] = a[i] + b[i]*(xxx-xt[i]) + c[i]*(xxx-xt[i])**2 + d[i]*(xxx-xt[i])**3

    return(q)

k = 3
h = 1/k
x = np.linspace(0, flower_new.shape[1], flower_new.shape[1]*k)
Iml = np.ndarray((flower_new.shape[0],flower_new.shape[1]*k,3))
# 2D Bilinear interpolation
for i in range(0, flower_new.shape[2]): # Work for each RGB matrix seperately
    x_t = flower_new[::, ::, i:i+1]
    
    #Interpolate horizontally:
    for j in range(0, len(x_t)): # Work for each row seperately
        xt = [i for i in range(0, len(x_t[0]))]
        yt = flower_new[j, :, i]
        Iml[j, :, i] = CubicSpline(xt,yt,x)
plt.imshow(Iml.astype(int))
        
Imll = np.ndarray((flower_new.shape[0]*k,flower_new.shape[1]*k,3))
x = np.linspace(0, flower_new.shape[0], flower_new.shape[0]*k)

for i in range(0, flower_new.shape[2]):
    x_t = flower_new[::, ::, i:i+1]
    
    #Interpolate vertically:
    for j in range(0, len(x_t[0])*k): # Work for each column seperately
        xt = [i for i in range(0, len(x_t))]
        yt = Iml[:, j, i]
        Imll[:, j, i] = CubicSpline(xt,yt,x)

plt.imshow(Imll.astype(int))

# Try with Lagrange, it's quite shite ;)

