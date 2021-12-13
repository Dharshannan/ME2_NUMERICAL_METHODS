#RK4 (Non-Matrix)
import numpy as np

h = 0.1
t = np.arange(0,0.5+h,h)
#initial y value
y_0 = 8
#Define function here:
def y_diff(t,y):
    y_t = (np.exp(-4*t))*(t*np.sin(t))*np.sqrt(10*y)
    return(y_t)

temp = y_0
for i in range(0,len(t)-1):
    k1 = h*y_diff(t[i],temp)
    k2 = h*y_diff(t[i]+0.5*h,temp + 0.5*k1)
    k3 = h*y_diff(t[i]+0.5*h,temp + 0.5*k2)
    k4 = h*y_diff(t[i]+h,temp + k3)
    yy = temp + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    temp = yy
    print([k1,k2,k3,k4,yy])
    
    

    
    
    

    
    