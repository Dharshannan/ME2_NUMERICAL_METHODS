# Laplace Dirichelt solution (Matrix_Method)
import numpy as np

# Dirichlet conditions, change this to fit any boundary conditions
h = 1 # Step size
x = np.arange(0,3+h,h) # Change x,y to fit the meshgrid of the required stencil
y = np.arange(0,3+h,h)
grid_points = np.zeros((len(y),len(x))) # Stencil of points
m,n = len(grid_points) - 1, len(grid_points[0]) - 1
for i in range(0,len(grid_points)):
    for j in range(0,len(grid_points[0])):
        # Define Dirichlet boundary conditions as stated in the question
        grid_points[0][j] = x[j]**4
        grid_points[i][n] = 81 - 54*y[i]**2 + y[i]**4
        grid_points[i][0] = y[i]**4
        grid_points[m][j] = x[j]**4 - 54*x[j]**2 + 81
        
# Points to evaluate:
eva = []
g = grid_points.copy()
#print(g)
for i in range(0,m):
    for j in range(0,n):
        if i != 0 and j != 0:
            eva.append((i,j))
#print(eva)         
# Number of points to evaluate:
k = len(eva)

M = np.zeros((k,k))
B = np.zeros(k)
sumb = []
a = 0
for i in range(1,m):
    for j in range(1,n):
        list1 = [(i+1,j), (i,j+1), (i-1,j), (i,j-1), (i,j)]
        sumb.append(list1)
        # Populate the B matrix
        if (i-1) == 0:
            B[a] -= g[i-1][j]
        if (i+1) == m:
            B[a] -= g[i+1][j]
        if (j-1) == 0:
            B[a] -= g[i][j-1]
        if (j+1) == n:
            B[a] -= g[i][j+1]
        
        a += 1
#print(B)
#print(sumb)
for i in range(0,k):
    for j in range(0,k):
        if eva[j] in sumb[i]:
            M[i][j] = 1
            
for k in range(0,k):
    M[k][k] = -4
    
#print(M)

'''sol = np.linalg.solve(M,B)

print(sol)'''

# Gauss Seidel with Relaxation:
u = [0,0,0,0] # initial guess (make sure it is the same size as M)
def Gauss(M,B,u,l,err): # l is lambda (relaxation factor), err is relative error in %
    Exit = 0
    n = 0 # Number of iterations
    while Exit != 1:
        temp = u.copy()
        for i in range(0, len(u)):
            sum_u = 0
            for j in range(0, len(u)):
                if j != i:
                    sum_u += M[i][j] * u[j]
            u[i] = (B[i] - sum_u)/M[i][i]
            u[i] = l*u[i] + (1-l)*temp[i]
        Pass = 0
        for j in range(0, len(u)):
            if abs((u[j] - temp[j])/(u[j]))*100 < err:
                Pass += 1
        if Pass == len(u):
            Exit = 1
        else:
            Exit = 0
        n += 1
        
    return(u,n)
           
U, N = Gauss(M,B,u,1.1,1) # set l as 1 for standard Guass Seidel
print('U Values:', U)
print('Number of iterations:', N)

list2 = list(zip(eva,U))
for i in range(0, len(list2)):
    x, y = list2[i][0]
    grid_points[x][y] = list2[i][1]
    
print(grid_points) # Prints out the meshgrid with the values of each point