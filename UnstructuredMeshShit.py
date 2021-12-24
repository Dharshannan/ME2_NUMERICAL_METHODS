import numpy as np
import matplotlib.pyplot as plt
# Barycentric interpolation function
def Barycen(rp,r,f):
    M = np.ndarray((3,3))
    B = np.ndarray((3,1))
    for i in range(0,3):
        M[0,i] = r[i,0]
        M[1,i] = r[i,1]
        M[2,i] = 1
        B[0] = rp[0]
        B[1] = rp[1]
        B[2] = 1
        
    l = np.linalg.solve(M,B)
    frp = l[0]*f[0] + l[1]*f[1] + l[2]*f[2]
    return(frp)
# Test function
'''r = np.array([[0,0],[0,1],[1,0]])
rp = np.array([0.5,0.5])
f = np.array([20,30,40])

print(Barycen(rp, r, f))'''

#Creating a circular unstructured mesh

R = 500 # Radius of the circle
L = 2 # Levels-Avicii ;)
N = 20 # Number of pointy triangle bois

rall = np.ndarray((N+1,2)) # All nodes
tri = np.zeros((N,3)) # All triangle bois, each containing 3 nodes
tri = tri.astype(int)
f = np.ndarray(N+1) # Func values for each node
r = np.ndarray((3,2)) # 3 Node per triangle
fr = np.ndarray(3) # 3 func per triangle

for i in range(0,N): # Set node values, match nodes to traingles, and set func values
    angle = i*(2*np.pi)/(N)
    rall[i,0] = R*np.cos(angle)
    rall[i,1] = R*np.sin(angle)
    tri[i,0] = i
    tri[i,1] = i+1
    tri[i,2] = N
    f[i] = i*(360/N)
    
rall[N] = [0,0] #Set final node at origin
tri[N-1,1] = 0 # Set final node for last triangle to be the 1st node (starting node)
f[N] =360 # Func at origin

for l in range(0,L):
    Nt = N
    ctri = np.copy(tri)
    N = Nt*3 # Number of triangles triple after every level
    tri = np.ndarray((N,3))
    tri = tri.astype(int)
    tricount = 0 # Counter for number of triangles
    Ng = len(rall)
    for j in range(0,Nt):
        for k in range(0,3): # Get nodes coordinates and func for each triangle
            r[k,:] = rall[ctri[j,k],:]
            fr[k] = f[ctri[j,k]]
        # Get center of triangle
        centrx = (r[0,0] + r[1,0] + r[2,0])/3
        centry = (r[0,1] + r[1,1] + r[2,1])/3
        
        rp = [centrx, centry]
        # Interpolate
        frp = Barycen(rp, r, fr)
        # Append new points to global set of nodes and func values
        rall = np.append(rall, [rp], axis=0)
        f = np.append(f,frp)
        # Baby triangle 1, new node located at the position Ng+j
        tri[tricount, :] = [ctri[j,0], ctri[j,1], Ng+j]
        # Baby triangle 2, new node located at the position Ng+j
        tri[tricount + 1, :] = [ctri[j,1], ctri[j,2], Ng+j]
        # Baby triangle 3, new node located at the position Ng+j
        tri[tricount + 2, :] = [ctri[j,2], ctri[j,0], Ng+j]
        tricount += 3 # Increment triangle counter by 3
        
# Plot some shit ;)
x = rall[:, 0]
y = rall[:, 1]
plt.scatter(x,y)
plt.axis('equal')
plt.show()


        
            
 



        
    
        
