import numpy as np

M = np.array([[1,0,0,0,0],[1,4,1,0,0],[0,1,4,1,0],[0,0,1,4,1],[0,0,0,0,1]], dtype=np.object)
b = np.array([0.0740,5.769,0,-5.769,-0.0740], dtype=np.object)
#print(M.shape)

def MyGauss(A,B):
    #A = A.astype('float64')
    #B = B.astype('float64')
    # i for columns
    for i in range(0, A.shape[1]):
        # j for rows
        for j in range(i, A.shape[0]-1):
            B[j+1] -= ((A[j+1][i])/(A[i][i]))*B[i]
            A[j+1] -= ((A[j+1][i])/(A[i][i]))*A[i]
    
    nrow = A.shape[0]
    x = np.zeros(nrow)
    #x = x.astype('float64')
    x[nrow-1] = B[nrow-1]/A[nrow-1][nrow-1]
    for i in range(nrow-2,-1,-1):
        itsum = 0
        for j in range(nrow-1, i,  -1):
            itsum += A[i][j]*x[j]
        x[i] = (B[i] - itsum)/A[i][i]

    return(x)

print(MyGauss(M,b))

