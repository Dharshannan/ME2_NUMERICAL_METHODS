import numpy as np
import matplotlib.pyplot as plt
# Crank-Nicolson is better, this direct method is shit!!!
# Cinosi's code is shit and inefficient, mine is better ;)
a = 1.172e-5
k = 1
h = 0.01
H = 500
K = 40
Tw = 5
t = np.arange(0,1200+k,k)
x = np.arange(0,0.5+h,h)
# populate 1D grid
grid = np.zeros(len(x))
for i in range(0,len(x)):
    if i == 0:
        grid[i] = 10
    elif i == len(x) - 1:
        grid[i] = 10
    elif i == ((len(x) - 1)/2):
        grid[i] = 100 # Heat generation
    else:
        grid[i] = 10
#print(grid)      
# Lets propagate this through time
for s in range(1,len(t)):       
    for i in range(0,len(x)):
        if i == 0:
            grid[i] = ((a*k)/(h**2))*(2*grid[i+1] - (2*h*H/K)*(grid[i] - Tw)) + (1 - ((2*a*k)/(h**2)))*(grid[i])
        if i == len(x) - 1:
            grid[i] = ((a*k)/(h**2))*(2*grid[i-1] - (2*h*H/K)*(grid[i] - Tw)) + (1 - ((2*a*k)/(h**2)))*(grid[i])
        else:
            grid[i] = ((a*k)/(h**2))*(grid[i+1] + grid[i-1]) + (1 - ((2*a*k)/(h**2)))*(grid[i])
        
    grid[int((len(x) - 1)/2)] = 100 # Heat generation

# Lets plot this shit mate, hell yeah!
plt.plot(x,grid)
plt.show()

    