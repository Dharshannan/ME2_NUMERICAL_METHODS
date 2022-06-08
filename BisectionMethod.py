import numpy as np 
import matplotlib.pyplot as plt

def Bisection(f, xu, xl, err):
    while True:
        xn = (xu+xl)/2
        if np.sign(f(xu))*np.sign(f(xn)) <= 0:
            xl = xn
        else:
            xu = xn
            
        if abs(xu-xn) <= err:
            break
    return(xn)
  
def f(x):
    return(x**2 + (x-2)**3 - 4)

root = Bisection(f, 1.85, 2.15, 1e-20)
print(root)