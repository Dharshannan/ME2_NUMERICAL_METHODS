# RK4 method 
import numpy as np
import matplotlib.pyplot as plt

h = 0.01
t = np.arange(0,60,h)
y_0 = np.array([0.2,0])
# Manipulate this matrix M and y_0(initial conditions), it is general hence can be used for any order *LINEAR SYSTEMS
M = np.array([[0,1],[-9,-0.1]])

def RK4(A,y_0):
    temp = y_0
    for j in range(0, len(y_0)):
       y_poop = []
       for k in range(len(y_0)):
           y_poop.append([])
           
       for n in range(0,len(y_0)):
           y_poop[n].append(y_0[n])
       temp = y_0   
       for i in range(0, len(t)-1):
           k1 = h*np.matmul(A,temp)
           k2 = h*np.matmul(A,temp + 0.5*k1)
           k3 = h*np.matmul(A,temp + 0.5*k2)
           k4 = h*np.matmul(A,temp + k3)
           y_t = temp + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        
           temp = y_t
           for j in range(0, len(y_0)):
               y_poop[j].append(temp[j])
    return(y_poop)
        
for i in range(0,len(y_0)):
    plt.plot(t,RK4(M, y_0)[i])
plt.show()







