import numpy as np 
import matplotlib.pyplot as plt

def f(x):
    return(x**2 + (x-2)**3 - 4)

def diff_f(x):
    return(2*x + 3*(x-2)**2)

def NewtonRaphson(xn, f, diff_f, err):
    temp = xn
    while True:
        x_new = temp - (f(temp)/diff_f(temp))
        if abs(x_new - temp) <= err:
            break
        temp = np.copy(x_new)
    return(temp) 

root = NewtonRaphson(1, f, diff_f, 1e-2)
print(root)

