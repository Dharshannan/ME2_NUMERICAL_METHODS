import numpy as np
import matplotlib.pyplot as plt 

h = 0.01
t = np.arange(0,60,h)
y_0 = [0.2,0]
A = np.array([[0,1],[-9,-0.2]])
I = np.identity(np.shape(A)[0])

y_diff = np.zeros((2,1))
temp = y_0
y_1 = [y_0[0]]
y_2 = [y_0[1]]
temp1 = y_0[0]
temp2 = y_0[1]
E_0 = (0.5*0**2) + (0.5*10*0.2**2)
E = [E_0]
for i in range(0, len(t)-1):
    
    # Total Energy
    y_t2 = temp2 + A[1][0]*h*(temp1) + A[1][1]*h*(temp2)
    temp2 = y_t2
    y_t1 =  temp1 + h*temp2
    temp1 = y_t1
    E_t = (0.5*y_t2**2) + (0.5*10*y_t1**2)

    y_1.append(temp1)
    y_2.append(temp2)
    E.append(E_t)
    
    '''# Forward Euler
    y_t = temp + h*np.matmul(A,temp)
    # Backward Euler
    #y_t = np.matmul((np.linalg.inv(I-h*A)),temp)
    temp = y_t
    E_t = (0.5*temp[1]**2) + (0.5*10*temp[0]**2)
    
    y_1.append(temp[0])
    y_2.append(temp[1])
    E.append(E_t)'''
    


plt.plot(t,y_1)
plt.plot(t,y_2)
#plt.plot(t,E)
#plt.plot(y_1,y_2)
plt.show()



