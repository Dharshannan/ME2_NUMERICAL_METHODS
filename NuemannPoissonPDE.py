import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
# Lets define both Nuemann and Dirichlet conditions
# Poisson : u(xx) + u(yy) = xy
h = 0.2 # Step size
x = np.arange(0,3+h,h) # Change x,y to fit the meshgrid of the required stencil
y = np.arange(0,3+h,h)
grid_points = np.zeros((len(y),len(x))) # Stencil of points
m,n = len(grid_points) - 1, len(grid_points[0]) - 1
# Dirichlet
for i in range(0,len(grid_points)):
    for j in range(0,len(grid_points[0])):
        grid_points[0][j] = 0
        grid_points[m][j] = x[j]**2
        grid_points[i][n] = 0
#print(grid_points)
# Lets assume that Nuemann conditions can only apply at a max of 2 surfaces(Grid faces)
# Neumann (lets apply this at the left surface):
neumann = []
for i in range(0,len(grid_points)):
    for j in range(0,len(grid_points[0])):
        if (i != 0) and (i != m) and j == 0: # Only applying neumann at the left surface
            nue = (i,j) # i represents y axis, j rep x axis (I know this is weird but its just a coding convention :) )
            neumann.append(nue)
#print(neumann) # nuemann is the index of the surfaces with nuemann conditions
# Points to evaluate
eva = []
g = grid_points.copy()
for i in range(0,m):
    for j in range(0,n):
        if i != 0 and j != 0:
            eva.append((i,j))
non_nue = len(eva) # Number of points that are not Nuemann condition points
for item in neumann:
    eva.append(item)
   
k = len(eva)

M = np.zeros((k,k))
B = np.zeros(k)
a = 0
for i in range(1,m):
    for j in range(1,n):
        B[a] = (h**2)*x[j]*y[i] # Change this based of RHS Poisson Equation
        a += 1

sumb = []
b = 0
for i in range(1,m):
    for j in range(1,n):
        list1 = [(i+1,j), (i,j+1), (i-1,j), (i,j-1), (i,j)]
        sumb.append(list1)
        # Populate the B matrix
        if (i-1) == 0:
            B[b] -= g[i-1][j]
        if (i+1) == m:
            B[b] -= g[i+1][j]
        if (j-1) == 0:
            B[b] -= g[i][j-1]
        if (j+1) == n:
            B[b] -= g[i][j+1]
        
        b += 1
#print(B)
for i in range(0,non_nue):
    for j in range(0,k):
        if eva[j] in sumb[i]:
            M[i][j] = 1

# Lets solve for Nuemann:
c = non_nue
for i in range(0,len(neumann)):
    s, t = neumann[i]
    B[c] = (h**2)*x[t]*y[s] - 6*h*y[s] # Change this based of the RHS of Poisson equation as well and Nuemann condition
    nuenue = [(s,t),(s,t+1)]
    for j in range(0, k):
        if eva[j] in nuenue:
            M[c][j] = 2
        if (s+1,t) in neumann:
            M[c][non_nue+(i+1)] = 1
        if (s-1,t) in neumann:
            M[c][non_nue+(i-1)] = 1
    c += 1

# Diagonalise the matrix M with values of -4
for k in range(0,k):
    M[k][k] = -4

# Update B matrix again based off Dirichlet boundary conditions (# A more generalized approach)
d = non_nue
for i in range(1,m):
    list2 = [(i-1,0), (i+1,0)]
    if (i-1) == 0:
        B[d] -= g[i-1][0]
    if (i+1) == m:
        B[d] -= g[i+1][0]
    d += 1

#print(M)         
#print(B)

sol = np.linalg.solve(M,B)

#print(sol)

list2 = list(zip(eva,sol))
for i in range(0, len(list2)):
    p, q = list2[i][0]
    grid_points[p][q] = list2[i][1]
    
print(grid_points) # Prints out the meshgrid with the values of each point

# Lets surface plot this shit :)

X, Y = np.meshgrid(x,y)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, grid_points, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor = 'none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
#ax.view_init(30,35) # Change the view direction

# Notes:
# The boundary conditions can be changed around to fit anything, this is a somewhat general code
# for example, you can tweak around with the Dirichlet conditions by changing some lines of code 
# or even simply changing the values
# The Nuemann boundary condition here is only applied to 1 surface as this is mostly the case
# in heat transfer problems, this can also be tweaked and changed.