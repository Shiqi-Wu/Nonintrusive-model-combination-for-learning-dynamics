import numpy as np
from scipy.optimize import minimize, Bounds
from robotics_dym import *
from parameters import *

h = 5
tt = np.arange(0, 180, dlt_t)
N = len(tt)

Q = np.zeros((n_K,n_K))
Q[4,4] = 1e8
Q[5,5] = 0.01
Q[6,6] = 1e8
R = 0.1 * np.eye(2)

def MPCsolver(s0, s_des, model, dic, w, h, N, mpc_prob, x_scaler, w_scaler, u_scaler):
    s_traj = [s0]
    u_opt = []
    s0 = s0.reshape((1,-1))
    s0_scaled = x_scaler.transform(s0)
#     print(s0_scaled)
    s0_dic = dic.call(s0_scaled)
    s_des_scaled = x_scaler.transform(s_des[:,1:7])
    s_des_scaled_dic = dic.call(s_des_scaled).numpy()
    w_scaled = w_scaler.transform(w)
    for i in range(N - h):
        s_des_h = s_des_scaled_dic[i:i+h+1,:]
        w_h = w_scaled[i:i+h+1,:]
        u = mpc_prob(s0_dic, s_des_h, model, w_h, h)
        u_inverse = u_scaler.inverse_transform(np.reshape(u,(-1,2)))
        u_opt.append(u_inverse[0])
        s0 = dynamics_solver(s0.flatten(), u_inverse[0], w[i,:])
        s0 = s0.reshape((1, -1))
#         print(s0)
        s_traj.append(s0)
        s0_scaled = x_scaler.transform(s0.reshape((1,-1)))
        s0_dic = dic.call(s0_scaled)
        if i % 50 == 0:
            print('step:%d'%i)
    return u_opt, s_traj


def mpc_prob_1_scipy(s0, s_des, model, w, h):

    def cost_function(u_flat):
        u = u_flat.reshape(h, u_dim)
        s = np.zeros((h + 1, n_K))
        s[0] = s0
        cost = 0
        for t in range(h):
            w_t_reshaped = np.array([w[t]]) 
            s[t + 1] = model([s[t].reshape((1,-1)), np.column_stack((u[t].reshape((1,-1)), w[t].reshape(1,-1)))])
            cost += np.dot((s[t + 1] - s_des[t + 1]).T, np.dot(Q, s[t + 1] - s_des[t + 1])) + np.dot((u[t] - np.array([0.5,0])).T, np.dot(R, u[t]-np.array([0.5,0])))
        return cost
    # print(u_dim)
    u0 = np.random.rand(h * u_dim)  # Initial guess for u
    # Define bounds for each u[t, i]
    inf_1 = 0
    sup_1 = 1
    inf_2 = 0
    sup_2 = 1
    bounds = Bounds([inf_1, inf_2] * h, [sup_1, sup_2] * h)

    result = minimize(cost_function, u0, bounds=bounds, method='L-BFGS-B')
    u_opt = result.x[:u_dim]
    # print(result.fun)
    return u_opt

def mpc_prob_2_scipy(s0, s_des, model, w, h):

    def cost_function(u_flat):
        u = u_flat.reshape(h, u_dim)
        s = np.zeros((h + 1, n_K))
        s[0] = s0
        cost = 0
        for t in range(h):
            w_t_reshaped = np.array([w[t]]) 
            s[t + 1] = model([s[t].reshape((1,-1)), np.column_stack((u[t].reshape((1,-1)), w[t].reshape(1,-1)))])
            cost += np.dot((s[t + 1] - s_des[t + 1]).T, np.dot(Q, s[t + 1] - s_des[t + 1])) + np.dot((u[t] - np.array([0.5,0])).T, np.dot(R, u[t]-np.array([0.5,0])))
        return cost

    u0 = 0 * np.random.rand(h * u_dim)  # Initial guess for u
    
    # Define bounds for each u[t, i]
    inf_1 = 0
    sup_1 = 1
    inf_2 = 0
    sup_2 = 1
    bounds = Bounds([inf_1, inf_2] * h, [sup_1, sup_2] * h)

    result = minimize(cost_function, u0, bounds=bounds, method='L-BFGS-B')
    u_opt = result.x[:u_dim]
    # print(result.fun)
    return u_opt

def mpc_prob_3_scipy(s0, s_des, model, w, h):

    def cost_function(u_flat):
        u = u_flat.reshape(h, u_dim)
        s = np.zeros((h + 1, n_K))
        s[0] = s0
        cost = 0
        for t in range(h-1):
            w_t_reshaped = np.array([w[t]]) 
            s[t + 1] = model([s[t].reshape((1,-1)), u[t].reshape((1,-1)), w[t].reshape((1,-1))])
            cost += np.dot((s[t+1] - s_des[t+1]).T, np.dot(Q, s[t+1] - s_des[t+1])) + np.dot((u[t] - np.array([0.5,0])).T, np.dot(R, u[t]-np.array([0.5,0])))
        return cost

    u0 = np.random.rand(h * u_dim)  # Initial guess for u
    
    # Define bounds for each u[t, i]
    inf_1 = 0
    sup_1 = 1
    inf_2 = 0
    sup_2 = 1
    bounds = Bounds([inf_1, inf_2] * h, [sup_1, sup_2] * h)

    result = minimize(cost_function, u0, bounds=bounds, method='L-BFGS-B')
    u_opt = result.x[:u_dim]
    # print(result.fun)
    return u_opt

