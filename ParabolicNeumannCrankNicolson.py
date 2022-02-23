import numpy as np
import matplotlib.pyplot as plt

# Lets code for a simple Parabolic PDE : u(t)=u(xx)
# Lets implement Crank Nicolson, lets also add Neumann Boundary conditions
# We will solve this by populating a matrix of coeeficients

# Define time steps
h = 0.25
k = h**2
t = np.arange(0,2,k)
x = np.arange(0,1+h,h) # Width of bar

# Define intial and boundary values
u_init = np.ones(len(x))
u_init[len(x)-1] = 0
plt.plot(x,u_init)
# Points to evalute, including 1st points (Neumann Boundary point), excluding last point
eva = []
for i in range(0,len(u_init)-1):
    eva.append(i)
    
temp = u_init.copy()
# Lets start populating the matrix
for s in range(0,len(t)):
    n = len(eva)
    M = np.zeros((n,n))
    B = np.zeros(n)
    a = 0
    # Populate B
    for j in range(0,len(u_init)-1):
        if j == 0:
            B[a] = 2*(temp[j+1])
        else:
            B[a] = temp[j+1] + temp[j-1]
        a += 1
    
    # Populate M
    
    coeff = []
    b = 0      
    for i in range(0,len(u_init)-1):
        coeff.append([i, i+1, i-1])
        
        if i+1 == len(u_init)-1:
            B[b] += temp[i+1]
        b += 1
        
    for i in range(1,n):
        for j in range(0,n):
            if eva[j] in coeff[i]:
                M[i][j] = -1
                
    # Lets update M with Neumann point
    m = 1 # 1 indicating only 1 point is Neumann
    for i in range(0,m):
        M[0][0] = 4
        M[0][1] = -2
        
    for k in range(0,n):
        M[k][k] = 4

    # Solve and update the points values
    evapoints = np.linalg.solve(M,B)
    index = list(zip(eva,evapoints))
    for i in range(0, len(index)):
        p = index[i][0]
        temp[p] = index[i][1]
    u_new = temp
    # Lets plot this shit
    plt.plot(x,u_new)
plt.show()
    
print(u_new)