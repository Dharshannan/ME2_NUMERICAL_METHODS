import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

# Lets code parabolic PDE of u(t) = u(xx) + u(yy) for a laptop (we will work in cm)
H = 0.06 # Heat transfer coefficients [W/cm^2.K]
alpha = 89.2E-2 # Thermal diffusivity of Silicon @ 300K [cm^2/s]
K = 1.48 # Thermal conductivity of Silicon @ 300K [W/cm.K]
h = 0.5 # Space step
k = 0.125 # Time step
r = k*alpha/(h**2)
t = np.arange(k,20+k,k)
x = np.arange(0,24.5+h,h) # Width of domain(laptop) discretised into points
y = np.arange(0,24.5+h,h) # Height of domain(laptop) discretised
Tgen = 80 # CPU chip temperature
Tinf = 10 # Cooling liquid temperature
Tcool = 10 # Cooling fan temperature
# Populate an initial stencil
grid_points = np.zeros((len(y),len(x))) # Stencil of points
m,n = len(grid_points) - 1, len(grid_points[0]) - 1
# Dirichlet
for i in range(0,len(grid_points)):
    for j in range(0,len(grid_points[0])):
        grid_points[i][j] = 60
        grid_points[0][j] = 35
        grid_points[i][n] = 35
        grid_points[m][j] = 35
        grid_points[int(m/2)][int(n/2)] = Tgen
        grid_points[int(m/4)][int(n/2)] = Tcool
        grid_points[int(3*m/4)][int(n/2)] = Tcool
        
#print(grid_points)

# Initial plot
X, Y = np.meshgrid(x,y)
'''ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, grid_points, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor = 'none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
ax.set_zlim([10,100])
#ax.view_init(60,35)'''
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, grid_points, cmap='YlOrRd')
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_xlabel('x')
ax.set_ylabel('y')
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
    
# We will now iterate through the time steps and Crank-Nicolson this shit, hell yea mate!!!
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
        B[c] = r*(g[s+1][p] + g[s-1][p] + 2*g[s][p+1] + (4*h*H/K)*Tinf) + (2 - 4*r - (2*h*H/K)*r)*(g[s][p]) # Nuemann for which -k*dT/dn = h(T-Tinf)
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
        M[i][i] = (2 + 4*r + (2*h*H/K)*r)
        
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
        
    g[int(m/2)][int(n/2)] = 0.25*t[l] + Tgen # Reset the midpoint (chip) temperature to Tgen (constant heat gen) OR Linearly increasing heat generation
    g[int(m/4)][int(n/2)] = Tcool # Reset fan temperature model constant heat sink
    g[int(3*m/4)][int(n/2)] = Tcool # Reset fan temperature model constant heat sink
    grid_new = g
    # Lets plot this shit
    X, Y = np.meshgrid(x,y)
    '''ax = plt.axes(projection = '3d')
    ax.plot_surface(X, Y, grid_new, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor = 'none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f')
    ax.set_zlim([10,100])
    #ax.view_init(60,35)'''
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, grid_new, cmap='YlOrRd')
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title('time:' f'{t[l]}s')
    plt.show()
