import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# We are coding a Crank-Nicolson Method to solve the Schrodinger Wave equation for a wave packet propagating through a double slit
# Assume the wave particles are confined within an infinite square well
def psi0(x, y, x0, y0, sigma=0.5, k=15*np.pi):
    # Initial state
    return ((1/(2*np.pi*sigma**2))**(0.25))*np.exp(-1/4*((x-x0)**2 + (y-y0)**2)/sigma**2)*np.exp(1j*k*(x-x0)) # Gaussian Wave packet

def doubleSlit_interaction(psi, j0, j1, i0, i1, i2, i3):
    psi = np.asarray(psi) 
    # cancel the wave function inside the walls of the double slit.
    psi[0:i3, j0:j1] = 0
    psi[i2:i1,j0:j1] = 0
    psi[i0:,  j0:j1] = 0
    
    return psi

# Parameters
L = 5 # Well of width L
Dy = 0.05 # Spatial step size.
Dt = Dy**2/4 # Temporal step size.
Nx = int(L/Dy) + 1 # Number of points on the x axis.
Ny = int(L/Dy) + 1 # Number of points on the y axis.
Nt = 300 # Number of time steps.
r = -Dt/(2j*Dy**2) # Constant to simplify expressions.

# Initial position of the center of the Gaussian wave function.
x0 = L/5
y0 = L/2

# Parameters of the double slit.
w = 0.2 # Width of the walls of the double slit.
s = 0.8 # Separation between the edges of the slits.
a = 0.4 # Aperture of the slits.

# Indices that parameterize the double slit in the space of points.
# Horizontal axis.
j0 = int(1/(2*Dy)*(L-w)) # Left edge.
j1 = int(1/(2*Dy)*(L+w)) # Right edge.
# Vertical axis.
i0 = int(1/(2*Dy)*(L+s) + a/Dy) # Lower edge of the lower slit.
i1 = int(1/(2*Dy)*(L+s))        # Upper edge of the lower slit.
i2 = int(1/(2*Dy)*(L-s))        # Lower edge of the upper slit.
i3 = int(1/(2*Dy)*(L-s) - a/Dy) # Upper edge of the upper slit.

# Time to solve this shit

x = np.linspace(0, L, Ny-2)
y = np.linspace(0, L, Ny-2)
x, y = np.meshgrid(x, y)
psis = [] # To store the wave function at each time step.

psi = psi0(x, y, x0, y0) # initialise the wave function with the Gaussian.
psi[0,:] = psi[-1,:] = psi[:,0] = psi[:,-1] = 0 # The wave function equals 0 at the edges of the simulation box (infinite potential well).
psi = doubleSlit_interaction(psi, j0, j1, i0, i1, i2, i3) # Initial interaction with the double slit.
psis.append(np.copy(psi))
m, n = len(psi), len(psi[0])

# Lets Crank Nicoloson this shit mate, hell yea!!
eva = []  # points to evaluate
for i in range(0,m):
    for j in range(0,n):
        eva.append((i,j))
        
k = m*n # Number of points to evaluate
A = np.zeros((k,k), complex) # Coefficient matrix of the LHS of the discretised wave equation
M = np.zeros((k,k), complex) # Coefficient matrix of the RHS of the discretised wave equation

# Populate A and M
coeff = []
for i in range(0,m):
    for j in range(0,n):
        # append the index of psi on both LHS and RHS of the equation for each i,j values
        coeff.append([(i,j), (i+1,j), (i-1,j), (i,j+1), (i,j-1)])
      
for i in range(0,k):
    for j in range(0,k):
        if eva[j] in coeff[i]:
            # Populate the matrix of coefficients
            A[i][j] = -1*r
            M[i][j] =  1*r
            
for i in range(0,k):
    # Update the diagonal of the matrix of coefficients
    A[i][i] = (1 + 4*r)
    M[i][i] = (1 - 4*r)

#print(A)
#print(M)

#Use scipy sparse to convert the matrix to a compressed sparse column for faster calcultions
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

Asp = csc_matrix(A)

for s in range(1,Nt):     
    vec = psi.reshape((k)) # reshape the psi matrix into a column vector
    b = np.matmul(M,vec) # compute the RHS column b by matrix multiplying M and vec
    vec = spsolve(Asp,b) # Solve the matrix for psi values
    psi = vec.reshape((Nx-2,Ny-2)) # reshape new psi values from column to matrix form
    psi = doubleSlit_interaction(psi, j0, j1, i0, i1, i2, i3) # update the psi values with the interaction at the double slit walls
    psis.append(np.copy(psi))
    
# calculate the modulus of the wave function at each time step.
mod_psis = [] # storing the modulus of the wave function at each time step.
for wavefunc in psis:
    mod = np.absolute(wavefunc) # calculate the modulus of the wavefunc values that are complex.
    mod_psis.append(mod) # save the calculated modulus.

# We will now plot the interference of the wave energy after it passes through the double slit
mod_psis = np.array(mod_psis)
sumlist = []
maxE  = 0 # Initiate maximum energy value
for t in range(len(mod_psis)):
    sumlist.append(mod_psis[t,:,75]) # append the wave energy values at each time for a fixed value of x, and y from (0,L)
    if max(mod_psis[t,:,75]) > maxE: # find the maximum energy value
        maxE = max(mod_psis[t,:,75])
        
# Initial plot of the Energy values without animation
'''for i in range(len(sumlist)):
    plt.plot(np.linspace(0,L,Ny-2), sumlist[i])
plt.ylabel('Energy')
plt.xlabel('y')
plt.show()'''

## Animate this interference pattern ##
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, L), ylim=(0, maxE + 0.05))
plt.ylabel('Energy')
plt.xlabel('y')
line, = ax.plot([], [])

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, L, Ny-2)
    y = sumlist[i]
    line.set_data(x, y)
    return line,

# call the animator
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=np.arange(0,Nt,2),repeat=True, blit=0)

plt.show() # Only run either 1 of the animations at a time 
'''
## In case there is need to save memory ##
del psis
del M
del vec

## Animation the actual plot ## 
fig = plt.figure() # create the figure.
ax = fig.add_subplot(111, xlim=(0,L), ylim=(0,L)) # add the subplot to the figure.

img = ax.imshow(mod_psis[0], extent=[0,L,0,L], cmap=plt.get_cmap("hot"), vmin=0, vmax=np.max(mod_psis), zorder=1, interpolation="none") # here the modulus of the 2D wave function shall be represented.

# paint the walls of the double slit with rectangles.
wall_bottom = Rectangle((j0*Dy,0),     w, i3*Dy,      color="w", zorder=50) # (x0, y0), width, height
wall_middle = Rectangle((j0*Dy,i2*Dy), w, (i1-i2)*Dy, color="w", zorder=50)
wall_top    = Rectangle((j0*Dy,i0*Dy), w, i3*Dy,      color="w", zorder=50)

# add the rectangular patches to the plot.
ax.add_patch(wall_bottom)
ax.add_patch(wall_middle)
ax.add_patch(wall_top)

# define the animation function for FuncAnimation.

def animate(i):
    img.set_data(mod_psis[i]) # fill img with the modulus data of the wave function.
    img.set_zorder(1)
    return img,

anim = FuncAnimation(fig, animate, interval=1, frames=np.arange(0,Nt,2), repeat=True, blit=0)

cbar = fig.colorbar(img)
plt.show()'''

## Save the animation ##
'''f = r"D:\ME2MCP\PDEs\Anim.gif" 
writergif = animation.PillowWriter(fps=30) 
anim.save(f, writer=writergif)'''