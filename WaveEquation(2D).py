import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# We are coding a wave equation u(tt) = u(xx) + u(yy)
# Define parameters
size = 10  #size of the square plate
dx = 0.1
x = np.arange(0,size+dx,dx)
y = np.arange(0,size+dx,dx)
X,Y = np.meshgrid(x,y)
T = 10


r = 0.5 #r = cdt/dx
c = 1
dt = (r*dx)/c

# Define the wave displacement array
u = np.zeros((len(y),len(x)))
m, n = len(u) - 1, len(u[0]) - 1
next_u = u.copy()
prev_u = u.copy()

# Implement an explicit method to solve the PDE
t = 0
while t<T:
    u[0,:] = 0
    u[:,0] = 0
    u[m,:] = 0
    u[:,n] = 0
    
    t = t+dt
    prev_u = u.copy() # When updating lists use .copy() instead of just saying prev_u = u, this is because python suffers from a problem in parallel computing
    u = next_u.copy() # Same as stated above
    
    u[int(m/2)][int(n/2)] = (dt**2)*20*np.sin(30*np.pi*t/20)
    
    for i in range(1,m):
        for j in range(1,n):
            next_u[i][j] = 2*u[i][j] - prev_u[i][j] + (r**2)*(u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j])

    # Plot some shit!
    '''print(u)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, u)
    plt.show()'''
    fig = plt.figure(figsize = [12,8])
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X,Y,u, cmap = cm.coolwarm, vmin = -0.02, vmax = 0.02)
    ax.set_zlim([-0.04,0.04])
    plt.show()
        