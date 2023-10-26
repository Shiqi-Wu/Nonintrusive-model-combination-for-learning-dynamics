import numpy as np
import tensorflow as tf

from Cardiac_Electrophysiology import *

chi = 140  # mm^-1
C_m = 0.01  # uF/mm^2
M = [0.174 / (chi * C_m), 0.174 / (chi * C_m)]

ic = Cardiac_Electrophysiology_Hodgkin_Huxley_1952.default_initial_conditions()

dlt_x = 0.2
x_lft, x_rht = 0, 10
num_x = int((x_rht - x_lft)/dlt_x + 1)
v0 = ic['V'] * np.ones((num_x, num_x))
m0 = ic['m'] * np.ones((num_x, num_x))
h0 = ic['h'] * np.ones((num_x, num_x))
n0 = ic['n'] * np.ones((num_x, num_x))
x0 = np.array([v0,m0,h0,n0])
x0 = np.reshape(x0,(1,-1))
x1 = np.linspace(x_lft, x_rht, num_x)
v_bd = ic['V'] * np.ones((num_x))

t = np.linspace(0, 10, 50)
# t_test = np.linspace(0, 10, 50)

from scipy.integrate import odeint
fun = Cardiac_Electrophysiology_DataModel()
# sol = odeint(fun.BuildFunction, x0[0], t, args=(M, dlt_x, num_x, x1, x1))
sol = odeint(fun.TestFunction, x0[0], t, args=(M, dlt_x, num_x, x1, x1))

v, s = [], []
for i in range(np.shape(sol)[0]):
    v_c, s_c = fun.vs_reshape(sol[i], num_x)
    v.append(v_c)
    s.append(s_c)

# v, s = [], []
# x0 = x0[0]
# for i in range(len(t)):
#     v_c, s_c = fun.vs_reshape(x0, num_x)
#     v.append(v_c)
#     s.append(s_c)
#     x0 = x0 + (t[1] - t[0]) * fun.BuildFunction(x0, t[i], M, dlt_x, num_x, x1, x1)
    
# np.save("./data/v_sample_0416.npy",v)
# np.save("./data/s_sample_0416.npy",s)
np.save("./data/v_test_50.npy",v)
np.save("./data/s_test_50.npy",s)