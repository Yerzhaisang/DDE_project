import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

def f_func(x):
    return (np.abs(x+1)-np.abs(x-1))/2

def equation(Y, t, d):
    c=[1,2] #non-negative constant
    a=[[0.1,0.3],[0.2,0.4]] #weights of NN are initialized randomly
    b=[[1.1,1.3],[2.2,2.4]] #weights of NN are initialized randomly
    I=[1.2,2.4] #the external bias
    x,y = Y(t)
    xd,yd = Y(t-d)
    y1 = -c[0]*x+a[0][0]*f_func(x)+a[0][1]*f_func(y)+b[0][0]*f_func(xd)+b[0][1]*f_func(yd)+I[0]
    y2 = -c[1]*y+a[1][0]*f_func(x)+a[1][1]*f_func(y)+b[1][0]*f_func(xd)+b[1][1]*f_func(yd)+I[1]
    return [y1, y2]

def initial_history_func(t):
    return [1., -1.]


plt.figure(figsize=(15, 7), dpi=80);
plt.rcParams['font.size'] = 12;
fig, axs = plt.subplots(1, 1);
fig.tight_layout(rect=[0, 0, 2, 2], pad=3.0);

ts = np.linspace(0, 3, 2000);

ys = ddeint(equation, initial_history_func, ts, fargs=(0.5,));
axs.plot(ts, ys[:,0], color='red', linewidth=2, label='$x_1(t)$');
axs.plot(ts, ys[:,1], color='blue', linewidth=2, label='$x_2(t)$');
axs.set_title('$H_1(t)=1; H_2(t)=-1; d=0.5$', fontsize=20);
axs.legend(fontsize=20);

plt.tight_layout()
plt.savefig('graph.png');
