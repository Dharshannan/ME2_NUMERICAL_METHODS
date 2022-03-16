import numpy as np
import matplotlib.pyplot as pl

# Lets define the 3 1st order ODEs that define the Blasius solution
# f(function of eta/non_dimensional position)
def diff_f(g):
    f_diff = g
    return(f_diff)
# g(non dimensional velocity)
def diff_g(h):
    g_diff = h 
    return(g_diff)
# h(non dimensional acceleration)
def diff_h(f,h):
    h_diff = -0.5*f*h
    return(h_diff)

# We will implememnt the RK4 method, coz I am just better than Cinosi ;)

def RK4(t0, tend, f, g, he):
    h = (tend - t0)/1000 # step size
    t = np.arange(t0, tend+h, h) # forward steps
    temp1 = f
    temp2 = g
    temp3 = he
    F = [f]
    G = [g]
    H = [he]
    # we will now iterate through the steps
    for i in range(0, len(t) - 1):
        k1_1 = h * diff_f(temp2)
        k2_1 = h * diff_f(temp2)
        k3_1 = h * diff_f(temp2)
        k4_1 = h * diff_f(temp2)
        
        k1_2 = h * diff_g(temp3)
        k2_2 = h * diff_g(temp3)
        k3_2 = h * diff_g(temp3)
        k4_2 = h * diff_g(temp3)
        
        k1_3 = h * diff_h(temp1, temp3)
        k2_3 = h * diff_h(temp1, temp3 + 0.5*k1_3)
        k3_3 = h * diff_h(temp1, temp3 + 0.5*k2_3)
        k4_3 = h * diff_h(temp1, temp3 + k3_3)
        
        ft1 = temp1 + (1/6)*(k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
        temp1 = ft1
        
        gt1 = temp2 + (1/6)*(k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
        temp2 = gt1
        
        ht1 = temp3 + (1/6)*(k1_3 + 2*k2_3 + 2*k3_3 + k4_3)
        temp3 = ht1
        
        F.append(temp1)
        G.append(temp2)
        H.append(temp3)
    return(F, G, H, t) 

# Function for lagrange interpolation
def Lagrange(xt, yt, x):
    lag = 0
    for j in range(0,len(xt)):
        mul = 1
        for k in range(0,len(xt)):
            if k != j:
                mul *= (x-xt[k])/(xt[j]-xt[k])
        lag += yt[j]*mul
    return(lag)
       
# Define the parameters
t0 = 0
tend = 10
f0 = 0 # inital f(0)
g0 = 0 # initial f'(0)
h0 = 1 # guess f''(0), this will be updated
gold = 0 # if h(0) = 0, then g(inf) = 0
hold = 0
# Lets implement the shooting method
tol = 1e-5 # tolerance value for the error
itmax = 1000 # max number of iterations
error = 100 # set initial error as large
it = 0 # initialise the number of iteartions
ghlist = [(gold,hold)] # to hold g,h values for each iteration for n_th order polynomial interpolation
while (error > tol) and (it < itmax):
    it += 1 # increment number of iterations
    # Lets RK4 this shit mate, hell yea :)
    F, G, H, eta = RK4(t0, tend, f0, g0, h0)
    g_inf = 1 # actual value at step infinity required for g
    ghlist.append([G[-1],h0])
    # We can either interpolate linearly for each iteration
    '''g = np.array([gold,G[-1]])
    h = np.array([hold,h0])
    # Interpolate
    hint = Lagrange(g, h, g_inf)
    gold = G[-1]
    hold = h0
    h0 = hint'''
    # Or we can interpolate using an n_th order polynomial for each iteration (# This is faster)
    # Sort this list
    def take_first(elem):
        return(elem[0])
    ghsorted = np.array(sorted(ghlist, key=take_first))
    # Interpolate:
    h0 = Lagrange(ghsorted[:,0], ghsorted[:,1], g_inf)
    # Calculate the error
    error = abs(G[-1] - g_inf)
print("Iterations required:", it)

# function for trapezium rule integration
def trapz(xt,yt):
    n = len(xt)
    b = xt[n-1]
    a = xt[0]
    mid_sum = 0
    for i in range(1, n-2):
        mid_sum += yt[i]  
    trp = ((b-a)/(n-1))*(yt[0]*0.5 + yt[n-1]*0.5 + mid_sum)
    return(trp)

# plotting section
f = F  # displacement
u = G  # velocity
v = 0.5*(eta*u-f) #

# calculate thicknesses
diff = abs(u-0.99*max(u))
eta99 = diff.argmin()
# displacement thickness delta1 = int 0->infinity of 1-u(eta)
d1 = trapz(eta,max(u)-u)
# momentum thickness delta2 = int 0->infinity of u(eta)*(1-u(eta))
d2 = trapz(eta,u*(max(u)-u))
print('d99: '+str(eta[eta99])+'   d1: '+str(d1)+'   d2: '+str(d2))

# plot u and thicknesses
pl.plot(u,eta)
pl.grid()
pl.plot([0,max(u)],[eta[eta99],eta[eta99]])
pl.plot([0,max(u)],[d1,d1])
pl.plot([0,max(u)],[d2,d2])
pl.xlabel('normalised u')
pl.ylabel('eta')
pl.show()

# plot v
pl.plot(v,eta)
pl.grid()
pl.xlabel('normalised v')
pl.ylabel('eta')
pl.show()