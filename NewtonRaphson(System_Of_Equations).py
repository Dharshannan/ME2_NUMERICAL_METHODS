import numpy as np 
import matplotlib.pyplot as plt

def u(x,y):
    return(x**2 + y**2 - 16)
def v(x,y):
    return((x-4)**2 + (y-4)**2 - 8)
def diff_ux(x,y):
    return(2*x)
def diff_uy(x,y):
    return(2*y)
def diff_vx(x,y):
    return(2*(x-4))
def diff_vy(x,y):
    return(2*(y-4))


def NewRaph(xn, yn, err):
    x_new = np.zeros((2,1))
    tempx = xn
    tempy = yn
    while True:
        x_prev = np.array([[tempx],[tempy]])
        f_prev = np.array([[u(tempx,tempy)],[v(tempx,tempy)]])
        jacobi = np.array([[diff_ux(tempx,tempy), diff_uy(tempx,tempy)], [diff_vx(tempx,tempy), diff_vy(tempx,tempy)]])
        invjacobi = np.linalg.inv(jacobi)
        x_new = x_prev - np.matmul(invjacobi, f_prev)
        if abs(x_new[0,0] - tempx) <= err and abs(x_new[1,0] - tempy) <= err:
            break
        tempx = np.copy(x_new[0,0])
        tempy = np.copy(x_new[1,0])  
    return(x_new)
    
print(NewRaph(1, 4, 1))