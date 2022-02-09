# Lets solve this problem using Gauss Seidel w/o populating the matrix of coefficients
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
# Lets define both Nuemann and Dirichlet conditions
# Poisson : u(xx) + u(yy) = xy
h = 0.1 # Step size
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
# Neumann (lets apply this at the left surface):  (*you can apply Neumann at the right side as well with a little changes to the code)
# Lets define all points to be evaluated including Neumann boundary condition points
# Points to evaluate
eva = []
g = grid_points.copy()
for i in range(1,m):
    for j in range(0,n):
            eva.append((i,j))
            
#print(eva) # Print this to check the arrangement of index of values to be evaluated to make sure the guesses corresponds to this arrangement

N = len(eva)
u = np.zeros(N) # Initial guess of values
# lets update the meshgrid with the u guess values
list2 = list(zip(eva,u))
for i in range(0, len(list2)):
    s, t = list2[i][0]
    grid_points[s][t] = list2[i][1]
#print(grid_points)
G = grid_points.copy()

# Lets use this meshgrid to evaluate each point using gauss seidel
# GAUSS SEIDEL WITHOUT MATRIX POPULATION:
# Without error:  
'''for k in range(15): # Number of iterations
    for i in range(1,m):
        for j in range(0,n):
            if j == 0: # Nuemann boundary points
                G[i][j] = (2*G[i][j+1] + G[i-1][j] + G[i+1][j] - (h**2 * (x[j]*y[i]) - 6*h*y[i]))/4
            # Now for other points:
            else:
                G[i][j] = (G[i+1][j] + G[i-1][j] + G[i][j-1] + G[i][j+1] - h**2 * (x[j]*y[i]))/4

print(G)'''
# With error
err = 0.0001 # apparent error for gauss seidel (%)
u_prev = u.copy()
Exit = 0
itr = 0 # Number of iterations
while Exit != 1: # Iterate until error for each point has reached err tolerance specified
    Pass = 0
    u_new = []
    for i in range(1,m):
        for j in range(0,n):
            if j == 0: # Nuemann boundary points
                G[i][j] = (2*G[i][j+1] + G[i-1][j] + G[i+1][j] - (h**2 * (x[j]*y[i]) - 6*h*y[i]))/4
                u_new.append(G[i][j])
            # Now for other points:
            else:
                G[i][j] = (G[i+1][j] + G[i-1][j] + G[i][j-1] + G[i][j+1] - h**2 * (x[j]*y[i]))/4
                u_new.append(G[i][j])

    for i in range(0,len(u_new)):
        if ((u_new[i] - u_prev[i])/(u_new[i]))*100 < err:
            Pass += 1
    u_prev = u_new.copy()
    if Pass == len(u_new):
        Exit = 1
    else:
        Exit = 0
    itr += 1

print(G) # Prints out the meshgrid with the values of each point
#print(itr) # Prints number of iterations
# Lets surface plot this shit :)

X, Y = np.meshgrid(x,y)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, G, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor = 'none')
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