import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

# Lets code parabolic PDE of u(t) = u(xx) + u(yy)

h = 0.1
k = 0.01
r = k/(h**2)
t = np.arange(k,0.25+k,k)
x = np.arange(0,1.5+h,h) # Width of bar discretised into points
y = np.arange(0,1.5+h,h) # Height of bar discretised Keep x and y same shape
# Populate an initial stencil
grid_points = np.zeros((len(y),len(x))) # Stencil of points
m,n = len(grid_points) - 1, len(grid_points[0]) - 1
# Dirichlet
for i in range(0,len(grid_points)):
    for j in range(0,len(grid_points[0])):
        grid_points[i][0] = 1
# Initial plot
X, Y = np.meshgrid(x,y)
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, grid_points, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor = 'none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
plt.title('time:0s')
plt.show()       

# Lets code for a simple Dirichlet at all boundaries
eva = []  # points to evaluate
g = grid_points.copy() # temporary variable to hold all initial values of the stencil
for i in range(1,m):
    for j in range(1,n):
        eva.append((i,j))

for s in range(len(t)):         
    k = len(eva)
    M = np.zeros((k,k))
    B = np.zeros(k)
    
    # Populate B
    a = 0
    for i in range(1,m):
        for j in range(1,n): 
            B[a] = r*(g[i+1][j] + g[i-1][j] + g[i][j+1] + g[i][j-1]) + (2 - 4*r)*(g[i][j])
            a += 1
            
    coeff = []
    b = 0
    for i in range(1,m):
        for j in range(1,n):
            coeff.append([(i,j), (i+1,j), (i-1,j), (i,j+1), (i,j-1)])
            if i-1 == 0:
                B[b] += r*(g[i-1][j])
            if i+1 == m:
                B[b] += r*(g[i+1][j])
            if j-1 == 0:
                B[b] += r*(g[i][j-1])
            if j+1 == n:
                B[b] += r*(g[i][j+1])
            b += 1
            
    # Populate M
    for i in range(0,k):
        for j in range(0,k):
            if eva[j] in coeff[i]:
                M[i][j] = -1*r
                
    for i in range(0,k):
        M[i][i] = (2 + 4*r)
        
    # Solve and update the points values
    evapoints = np.linalg.solve(M,B)
    index = list(zip(eva,evapoints))
    for i in range(0, len(index)):
        p = index[i][0]
        g[p] = index[i][1]
    grid_new = g
    # Lets plot this shit
    X, Y = np.meshgrid(x,y)
    ax = plt.axes(projection = '3d')
    ax.plot_surface(X, Y, grid_new, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor = 'none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f')
    plt.title('time:' f'{t[s]}s')
    plt.show()       