import numpy as np


def MyGauss(A, I):
    A = A.astype('float64')
    # i for columns
    for i in range(0, np.shape(A)[1]):
        # j for rows
        for j in range(i, np.shape(A)[0]-1):
            I[j+1] -= ((A[j+1][i])/(A[i][i]))*I[i]
            A[j+1] -= ((A[j+1][i])/(A[i][i]))*A[i]
            
    #return(I)    
    for i in range(np.shape(A)[1]-1, -1, -1):
        # j for rows
        for j in range(i, 0, -1):
            I[j-1] -= ((A[j-1][i])/(A[i][i]))*I[i]
            A[j-1] -= ((A[j-1][i])/(A[i][i]))*A[i]
    #return(I)  
     # Diogonal formed
    for i in range(0, np.shape(A)[0]):
        I[i] = I[i]*(1/(A[i][i]))
        A[i] = A[i]*(1/(A[i][i]))

    return(I)

m = np.array([[-1,3,4],[9,0,1],[8,6,8]])
In = np.identity(np.shape(m)[0])
print(MyGauss(m,In))
real = np.linalg.inv(m)
print(real)

