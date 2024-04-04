import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

m1 = 1
m2 = 1
l1 = 1
l2 = 0.5
L = l1 + l2
g = 9.81
h = 0.002
t = np.arange(0,15,h)
history_len = 500 

def y_diff1(y_1,y_2,y_3,y_4,m1,m2,l1,l2,g):
    y1_diff = y_2
    return(y1_diff)

def y_diff2(y_1,y_2,y_3,y_4,m1,m2,l1,l2,g):
    y2_diff = (m2*g*np.sin(y_3)*np.cos(y_1-y_3) - m2*np.sin(y_1-y_3)*(l1*(y_2**2)*np.cos(y_1-y_3) + l2*(y_4**2)) - (m1+m2)*g*np.sin(y_1))/(l1*(m1 + m2*(np.sin(y_1-y_3))**2))
    return(y2_diff)

def y_diff3(y_1,y_2,y_3,y_4,m1,m2,l1,l2,g):
    y3_diff = y_4
    return(y3_diff)

def y_diff4(y_1,y_2,y_3,y_4,m1,m2,l1,l2,g):
    y4_diff = ((m1+m2)*(l1*(y_2**2)*np.sin(y_1-y_3) + g*np.sin(y_1)*np.cos(y_1-y_3) - g*np.sin(y_3)) + m2*l2*(y_4**2)*np.sin(y_1-y_3)*np.cos(y_1-y_3))/(l2*(m1 + m2*(np.sin(y_1-y_3))**2))
    return(y4_diff)

y_1 = np.radians(120)
y_2 = np.radians(0)
y_3 = np.radians(-20)
y_4 = np.radians(0)



def RK4():
    temp1 = y_1
    temp2 = y_2
    temp3 = y_3
    temp4 = y_4
    y1 = [y_1]
    y2 = [y_2]
    y3 = [y_3]
    y4 = [y_4]
    for i in range(0, len(t)-1):
# =============================================================================
#         ## Wrong Implementation ##
#         k1_1 = h*y_diff1(temp1, temp2, temp3, temp4, m1, m2, l1, l2, g)
#         k2_1 = h*y_diff1(temp1 + 0.5*k1_1, temp2, temp3, temp4, m1, m2, l1, l2, g)
#         k3_1 = h*y_diff1(temp1 + 0.5*k2_1, temp2, temp3, temp4, m1, m2, l1, l2, g)
#         k4_1 = h*y_diff1(temp1 + k3_1, temp2, temp3, temp4, m1, m2, l1, l2, g)
#         
#         k1_2 = h*y_diff2(temp1, temp2, temp3, temp4, m1, m2, l1, l2, g)
#         k2_2 = h*y_diff2(temp1, temp2 + 0.5*k1_2, temp3, temp4, m1, m2, l1, l2, g)
#         k3_2 = h*y_diff2(temp1, temp2 + 0.5*k2_2, temp3, temp4, m1, m2, l1, l2, g)
#         k4_2 = h*y_diff2(temp1, temp2 + k3_2, temp3, temp4, m1, m2, l1, l2, g)
#         
#         k1_3 = h*y_diff3(temp1, temp2, temp3, temp4, m1, m2, l1, l2, g)
#         k2_3 = h*y_diff3(temp1, temp2, temp3 + 0.5*k1_3, temp4, m1, m2, l1, l2, g)
#         k3_3 = h*y_diff3(temp1, temp2, temp3 + 0.5*k2_3, temp4, m1, m2, l1, l2, g)
#         k4_3 = h*y_diff3(temp1, temp2, temp3 + k3_3, temp4, m1, m2, l1, l2, g)
#         
#         k1_4 = h*y_diff4(temp1, temp2, temp3, temp4, m1, m2, l1, l2, g)
#         k2_4 = h*y_diff4(temp1, temp2, temp3, temp4 + 0.5*k1_4, m1, m2, l1, l2, g)
#         k3_4 = h*y_diff4(temp1, temp2, temp3, temp4 + 0.5*k2_4, m1, m2, l1, l2, g)
#         k4_4 = h*y_diff4(temp1, temp2, temp3, temp4 + k3_4, m1, m2, l1, l2, g)
# =============================================================================
        ## Correct Implementation ##
        k1_1 = h*y_diff1(temp1, temp2, temp3, temp4, m1, m2, l1, l2, g)
        k1_2 = h*y_diff2(temp1, temp2, temp3, temp4, m1, m2, l1, l2, g)
        k1_3 = h*y_diff3(temp1, temp2, temp3, temp4, m1, m2, l1, l2, g)
        k1_4 = h*y_diff4(temp1, temp2, temp3, temp4, m1, m2, l1, l2, g)
        
        k2_1 = h*y_diff1(temp1 + 0.5*k1_1, temp2 + 0.5*k1_2, temp3 + 0.5*k1_3, temp4 + 0.5*k1_4, m1, m2, l1, l2, g)
        k2_2 = h*y_diff2(temp1 + 0.5*k1_1, temp2 + 0.5*k1_2, temp3 + 0.5*k1_3, temp4 + 0.5*k1_4, m1, m2, l1, l2, g)
        k2_3 = h*y_diff3(temp1 + 0.5*k1_1, temp2 + 0.5*k1_2, temp3 + 0.5*k1_3, temp4 + 0.5*k1_4, m1, m2, l1, l2, g)
        k2_4 = h*y_diff4(temp1 + 0.5*k1_1, temp2 + 0.5*k1_2, temp3 + 0.5*k1_3, temp4 + 0.5*k1_4, m1, m2, l1, l2, g)
        
        k3_1 = h*y_diff1(temp1 + 0.5*k2_1, temp2 + 0.5*k2_2, temp3 + 0.5*k2_3, temp4 + 0.5*k2_4, m1, m2, l1, l2, g)
        k3_2 = h*y_diff2(temp1 + 0.5*k2_1, temp2 + 0.5*k2_2, temp3 + 0.5*k2_3, temp4 + 0.5*k2_4, m1, m2, l1, l2, g)
        k3_3 = h*y_diff3(temp1 + 0.5*k2_1, temp2 + 0.5*k2_2, temp3 + 0.5*k2_3, temp4 + 0.5*k2_4, m1, m2, l1, l2, g)
        k3_4 = h*y_diff4(temp1 + 0.5*k2_1, temp2 + 0.5*k2_2, temp3 + 0.5*k2_3, temp4 + 0.5*k2_4, m1, m2, l1, l2, g)
        
        k4_1 = h*y_diff1(temp1 + k3_1, temp2 + k3_2, temp3 + k3_3, temp4 + k3_4, m1, m2, l1, l2, g)
        k4_2 = h*y_diff2(temp1 + k3_1, temp2 + k3_2, temp3 + k3_3, temp4 + k3_4, m1, m2, l1, l2, g)
        k4_3 = h*y_diff3(temp1 + k3_1, temp2 + k3_2, temp3 + k3_3, temp4 + k3_4, m1, m2, l1, l2, g)
        k4_4 = h*y_diff4(temp1 + k3_1, temp2 + k3_2, temp3 + k3_3, temp4 + k3_4, m1, m2, l1, l2, g)
        
        y_t1 = temp1 + (1/6)*(k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
        temp1 = y_t1
        y_t2 = temp2 + (1/6)*(k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
        temp2 = y_t2
        y_t3 = temp3 + (1/6)*(k1_3 + 2*k2_3 + 2*k3_3 + k4_3)
        temp3 = y_t3
        y_t4 = temp4 + (1/6)*(k1_4 + 2*k2_4 + 2*k3_4 + k4_4)
        temp4 = y_t4
        y1.append(temp1)
        y3.append(temp3)
    return(y1,y3)
              
x1 = l1*np.sin(np.array(RK4()[0]))
y1 = -l1*np.cos(np.array(RK4()[0]))

x2 = l2*np.sin(np.array(RK4()[1])) + x1
y2 = -l2*np.cos(np.array(RK4()[1])) + y1

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*h))
    return line, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y1), interval=h*1000, blit=True)
plt.show()