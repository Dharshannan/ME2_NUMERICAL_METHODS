import numpy as np
# Compound simpson
def simp(x,y):
    N = len(x) - 1
    h = x[1] - x[0]
    # Odd sum node
    sumodd = 0
    sumeven = 0
    for i in range(1,N,2):
        sumodd += y[i]
    # Even sum node
    for j in range(2,N,2):
        sumeven += y[j]
        
    simps = (h/3)*(y[0] + y[N] + 4*sumodd + 2*sumeven)
    return(simps)
# l starts with 2, as the min spacing is 2
l = 2
tol = 10**(-13)
while True:
    h = (np.pi)/l
    x = np.arange(0,np.pi+h,h)
    y = np.sin(x)
    I1 = simp(x,y)
    l *= 2 #multiply l by 2 to half step size
    h2 = (np.pi)/l
    x2 = np.arange(0,np.pi+h2,h2)
    y2 = np.sin(x2)
    #print(h,h2)
    
    I2 = simp(x2,y2)
    error = (1/15)*(I2 - I1) # Crude error 
    if abs(error) <= tol: #if the tolerance is achieved exit the loop
        break
    
print(I1,I2,error)




        
        