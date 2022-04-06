import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

# Lets code parabolic PDE of u(t) = u(xx) + u(yy)

h = 0.1
k = 0.01
r = k/(h**2)
t = np.arange(k,1+k,k)
x = np.arange(0,3+h,h) # Width of domain discretised into points
y = np.arange(0,3+h,h) # Height of domain discretised
# Populate an initial stencil
grid_points = np.zeros((len(y),len(x))) # Stencil of points
m,n = len(grid_points) - 1, len(grid_points[0]) - 1
# Dirichlet
for i in range(0,len(grid_points)):
    for j in range(0,len(grid_points[0])):
        grid_points[0][j] = 0
        grid_points[m][j] = x[j]**2
        grid_points[i][n] = 0
#print(grid_points)

# Initial plot
X, Y = np.meshgrid(x,y)
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, grid_points, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor = 'none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
plt.title('time:0s')
plt.show()

# Lets code for a Dirichlet at 3 boundaries and Nuemann at the left hand side boundary
eva = []  # points to evaluate excluding Nuemann points
g = grid_points.copy() # temporary variable to hold all initial values of the stencil
for i in range(1,m):
    for j in range(1,n):
        eva.append((i,j))
        
# Neumann (lets apply this at the left surface):
neumann = []
for i in range(0,len(grid_points)):
    for j in range(0,len(grid_points[0])):
        if (i != 0) and (i != m) and j == 0: # Only applying neumann at the left surface
            nue = (i,j) # i represents y axis, j rep x axis (I know this is weird but its just a coding convention :) )
            neumann.append(nue)

non_nue = len(eva) # Number of points that are not Nuemann condition points
for item in neumann:
    eva.append(item) # now we include the nuemann points in the points to evaluate
     
for l in range(len(t)):         
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
            if j+1 == n:
                B[b] += r*(g[i][j+1])
            b += 1
            
    # Populate M
    for i in range(0,non_nue):
        for j in range(0,k):
            if eva[j] in coeff[i]:
                M[i][j] = -1*r
            
    # Lets solve for Nuemann:
    c = non_nue
    for i in range(0,len(neumann)):
        s, p = neumann[i]
        B[c] = r*(g[s+1][p] + g[s-1][p] + 2*g[s][p+1] + 12*h*y[s]) + (2 - 4*r)*(g[s][p]) # Nuemann for which du/dn = 3*y
        nuenue = [(s,p),(s,p+1)]
        for j in range(0, k):
            if eva[j] in nuenue:
                M[c][j] = -2*r
            if (s+1,p) in neumann:
                M[c][non_nue+(i+1)] = -1*r
            if (s-1,p) in neumann:
                M[c][non_nue+(i-1)] = -1*r
        c += 1
                
    for i in range(0,k):
        M[i][i] = (2 + 4*r)
        
    # Update B matrix again based off Dirichlet boundary conditions (# A more generalized approach)
    d = non_nue
    for i in range(1,m):
        list2 = [(i-1,0), (i+1,0)]
        if (i-1) == 0:
            B[d] += r*g[i-1][0]
        if (i+1) == m:
            B[d] += r*g[i+1][0]
        d += 1
        
    # Solve and update the points values
    sol = np.linalg.solve(M,B)
    list2 = list(zip(eva,sol))
    for i in range(0, len(list2)):
        w, q = list2[i][0]
        g[w][q] = list2[i][1]
    grid_new = g
    # Lets plot this shit
    X, Y = np.meshgrid(x,y)
    ax = plt.axes(projection = '3d')
    ax.plot_surface(X, Y, grid_new, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor = 'none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f')
    plt.title('time:' f'{t[l]}s')
    plt.show()