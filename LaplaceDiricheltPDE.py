# Laplace Dirichelt solution (Matrix_Method)
import numpy as np

# Dirichlet conditions, change this to fit any boundary conditions
grid_points = np.zeros((5,5)) # Stencil of points
m,n = len(grid_points) - 1, len(grid_points[0]) - 1
for i in range(0,len(grid_points)):
    for j in range(0,len(grid_points[0])):
        
        grid_points[i][0] = 1
        
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

sol = np.linalg.solve(M,B)

#print(sol)

list2 = list(zip(eva,sol))
for i in range(0, len(list2)):
    x, y = list2[i][0]
    grid_points[x][y] = list2[i][1]
    
print(grid_points) # Prints out the meshgrid with the values of each point
        