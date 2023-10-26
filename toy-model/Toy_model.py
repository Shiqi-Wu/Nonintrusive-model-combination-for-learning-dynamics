import numpy as np
from scipy.optimize import minimize

class Toymodel(object):
    def __init__(self, dim = 20, L0 = 0, L1 = 1, mu = 1):
        self.dim = dim
        self.L0 = L0
        self.L1 = L1
        self.mu = mu
    
    def reaction_term(self, u):
        return 1/4*2 * (u-1)*(u+1)**2+ 1/4 * 2 * (u-1)**2 * (u+1)
    
    def build_A(self):
        self.A = np.zeros([self.dim - 1, self.dim - 1])
        for i in range(self.dim - 1):
            self.A[i, i] = -2 
        for i in range(self.dim - 2):
            self.A[i, i + 1] = 1
            self.A[i + 1, i] = 1
    
    def generate_traj(self, steps, u0, dlt_t = 0.001):
        u_data = np.zeros([steps + 1, self.dim - 1])
        u_data[0, :] = u0
        dlt_x = (self.L1 - self.L0)/self.dim
        for step in range(steps):
            u_0 = u_data[step, :]
            u_1 = u_0 + dlt_t * (self.mu * u_0 @ self.A/ dlt_x ** 2) + self.reaction_term(u_0)
            u_data[step + 1,:] = u_1
        return u_data
    
    def genarate_training_data(self, steps, traj_num, dlt_t = 0.001):
        u0 = np.random.rand(self.dim - 1)
        u_data = self.generate_traj(steps, u0, dlt_t)
        u_x = u_data[:-1,:]
        u_y = u_data[1:,:]
        for i in range(traj_num - 1):
            u0 = np.random.rand(self.dim - 1)
            u = self.generate_traj(steps, u0, dlt_t)
            u_data = np.concatenate((u_data, u), axis = 0)
            u_x = np.concatenate((u_x, u[:-1,:]), axis = 0)
            u_y = np.concatenate((u_y, u[1:,:]), axis = 0)
        u_x_lace = u_x @ self.A
        data_x = np.reshape(u_x, (-1, 1))
        data_y = np.reshape(u_y, (-1, 1))
        data_lace = np.reshape(u_x_lace, (-1,1))
        return data_x, data_y, data_lace
    
class DicPoly(object):
    def __init__(self, degree = 2):
        self.dictionary_degree = degree
    
    def call(self, data):
        poly = []
        poly_index = []
        pre_poly = []
        N = np.shape(data)[1]
        for j in range(N):
            pre_poly.append([j])
        for i in range(1, self.dictionary_degree):
            cur_poly = []
            for pp in pre_poly:
                index = pp[-1]
                for j in range(index, N):
                    cur_term = pp[:]
                    cur_term.append(j)
                    cur_poly.append(cur_term)
                    poly_index.append(cur_term)
            pre_poly = cur_poly
        for index in poly_index:
            cur_poly_data = np.ones(np.shape(data)[0])
            for i in index:
                cur_poly_data = cur_poly_data * data[:,i]
            poly.append(cur_poly_data)
        poly = np.array(poly)
        poly = np.transpose(poly)
        ones = np.ones((np.shape(data)[0], 1))
        return np.concatenate((ones, data, poly),axis=1)
    
def Hybrid_compute_linear_weight(x,y):
    y = np.reshape(y,(-1,1))
    A = np.matmul(np.transpose(x),x)
    b = np.matmul(np.transpose(x),y)
    w = np.matmul(np.linalg.inv(A),b)
    y_pred = np.matmul(x,w)
    return w, y_pred

def Original_Hybrid_Method(x_data, y_data, x_lace, dic, mu_error_bound):
    """
    x_data, y_data: dataset
    """
    dic_x = dic.call(x_data)
    mu_current, mu_pred = 0, 1e-3
    mu_history = []
    err_history = []
    koopman_history = []
    y_linear = y_data
    inverse_matrix = np.linalg.inv(dic_x.T @ dic_x)
    epoch = 0

    while np.linalg.norm(mu_current - mu_pred) > mu_error_bound:
        mu_current = mu_pred
        mu_pred, y_linear_pred = Hybrid_compute_linear_weight(x_lace,y_linear)
        y_koopman = y_data - y_linear_pred
        dic_y_koopman = dic.call(y_koopman)
        K = inverse_matrix @ dic_x.T @ dic_y_koopman
        dic_y_koopman_pred = dic_x @ K
        y_koopman_pred = np.reshape(dic_y_koopman_pred[:,1],(-1,1))
        y_linear = y_data - y_koopman_pred
        mu_history.append(mu_pred)
        err_history.append(np.linalg.norm(y_linear - y_linear_pred, ord=np.inf))
        koopman_history.append(y_koopman_pred)
        print('epoch = %d, the error is %f' %(epoch, err_history[-1]))
        epoch = epoch + 1
    
    return mu_pred, K, mu_history, err_history, koopman_history

def Relaxed_Hybrid_Method(x_data, y_data, x_lace, dic, mu_error_bound):
    """
    x_data, y_data: dataset
    """
    dic_x = dic.call(x_data)
    mu_current = [0]
    mu_history = []
    err_history = []
    koopman_history = []
    data_linear = y_data
    y_linear = y_data
    inverse_matrix = np.linalg.inv(dic_x.T @ dic_x)

    # First step
    mu_pred, y_linear_pred = Hybrid_compute_linear_weight(x_lace,y_linear)
    y_koopman = y_data - y_linear_pred
    dic_y_koopman = dic.call(y_koopman)
    K = np.linalg.inv(dic_x.T @ dic_x) @ dic_x.T @ dic_y_koopman
    dic_y_koopman_pred = dic_x @ K
    y_koopman_pred = np.reshape(dic_y_koopman_pred[:,1],(-1,1))
    y_linear = y_data - y_koopman_pred
    mu_history.append(mu_pred)
    err_history.append(np.linalg.norm(y_linear - y_linear_pred, ord=np.inf))
    koopman_history.append(y_koopman_pred)
    while np.linalg.norm(mu_current - mu_pred) > mu_error_bound:
        y_linear_pred_pre = y_linear_pred
        y_koopman_pred_pre = y_koopman_pred
        mu_current[:] = mu_pred[:]
        mu_pred, y_linear_pred = Hybrid_compute_linear_weight(x_lace,y_linear)
        y_koopman = y_data - y_linear_pred
        dic_y_koopman = dic.call(y_koopman)
        K = np.linalg.inv(dic_x.T @ dic_x) @ dic_x.T @ dic_y_koopman
        dic_y_koopman_pred = dic_x @ K
        y_koopman_pred = np.reshape(dic_y_koopman_pred[:,1],(-1,1))
        y_linear = y_data - y_koopman_pred
        
        t_F = np.transpose(y_data - y_linear_pred_pre - y_koopman_pred_pre) @ (y_linear_pred + y_koopman_pred - y_linear_pred_pre - y_koopman_pred_pre)/(np.transpose(y_linear_pred + y_koopman_pred - y_linear_pred_pre - y_koopman_pred_pre) @ (y_linear_pred + y_koopman_pred - y_linear_pred_pre - y_koopman_pred_pre))
        y_linear = y_data - (t_F * y_koopman_pred + (1 - t_F) * y_koopman_pred_pre)
        mu_history.append(mu_pred)
        err_history.append(np.linalg.norm(y_linear - y_linear_pred, ord=np.inf))
        koopman_history.append(y_koopman_pred)
    return mu_pred, K, mu_history, err_history, koopman_history

def Reflected_Hybrid_Method(x_data, y_data, x_lace, dic, mu_error_bound):
    """
    x_data, y_data: dataset
    """
    dic_x = dic.call(x_data)
    mu_current = [0]
    mu_history = []
    err_history = []
    koopman_history = []
    data_linear = y_data
    y_linear = y_data
    inverse_matrix = np.linalg.inv(dic_x.T @ dic_x)

    # First step
    mu_pred, y_linear_pred = Hybrid_compute_linear_weight(x_lace,y_linear)
    y_koopman = y_data - y_linear_pred
    dic_y_koopman = dic.call(y_koopman)
    K = np.linalg.inv(dic_x.T @ dic_x) @ dic_x.T @ dic_y_koopman
    dic_y_koopman_pred = dic_x @ K
    y_koopman_pred = np.reshape(dic_y_koopman_pred[:,1],(-1,1))
    y_linear = y_data - y_koopman_pred
    mu_history.append(mu_pred)
    err_history.append(np.linalg.norm(y_linear - y_linear_pred, ord=np.inf))
    koopman_history.append(y_koopman_pred)
    while np.linalg.norm(mu_current - mu_pred) > mu_error_bound:
        y_linear_pred_pre = y_linear_pred
        y_koopman_pred_pre = y_koopman_pred
        mu_current[:] = mu_pred[:]
        mu_pred, y_linear_pred = Hybrid_compute_linear_weight(x_lace,y_linear)
        y_koopman = y_data - y_linear_pred
        dic_y_koopman = dic.call(y_koopman)
        K = np.linalg.inv(dic_x.T @ dic_x) @ dic_x.T @ dic_y_koopman
        dic_y_koopman_pred = dic_x @ K
        y_koopman_pred = np.reshape(dic_y_koopman_pred[:,1],(-1,1))
        y_linear = y_data - y_koopman_pred
        
        t_F = 2
        y_linear = y_data - (t_F * y_koopman_pred + (1 - t_F) * y_koopman_pred_pre)
        mu_history.append(mu_pred)
        err_history.append(np.linalg.norm(y_linear - y_linear_pred, ord=np.inf))
        koopman_history.append(y_koopman_pred)
    return mu_pred, K, mu_history, err_history, koopman_history

def Hybrid_predictor(mu, K, dic, u0, steps):
    dim = len(u0)
    A = np.zeros([dim, dim])
    for i in range(dim):
        A[i, i] = -2 
    for i in range(dim - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    u_pred = np.zeros([steps + 1, dim])
    u_pred[0,:] = u0
    for i in range(steps):
        u0 = u_pred[i,:]
        u_linear = mu * u0 @ A
        u_koopman = dic.call(np.reshape(u0,(-1,1))) @ K
        u1 = u_linear + u_koopman[:,1]
        u_pred[i+1,:] = u1
    return u_pred
        
def Koopman_solver(x_data, y_data, dic):
    dic_x = dic.call(x_data)
    dic_y = dic.call(y_data)
    return np.linalg.inv(dic_x.T @ dic_x) @ dic_x.T @ dic_y

def Koopman_predictor(K, dic, u0, steps, state_dim=1):
    dim = len(u0)
    u_pred = np.zeros([steps + 1, dim])
    u_pred[0,:] = u0
    for i in range(steps):
        u0 = u_pred[i,:]
        u_koopman = dic.call(np.reshape(u0,(-1,state_dim))) @ K
        u1 = u_koopman[:,1:state_dim+1]
        u_pred[i+1,:] = np.reshape(u1,(1,-1))[0]
    return u_pred
    
    
class MethodCriterion(object):
    def __init__(self, dim = 20, L0 = 0, L1 = 1, mu = [1,1]):
        self.dim = dim
        self.L0 = L0
        self.L1 = L1
        self.mu = mu
        
    def build_A(self):
        self.A = np.zeros([self.dim - 1, self.dim - 1])
        for i in range(self.dim - 1):
            self.A[i, i] = -2 
        for i in range(self.dim - 2):
            self.A[i, i + 1] = 1
            self.A[i + 1, i] = 1
    
    def generate_traj(self, steps, u0, dlt_t = 0.001):
        u_data = [u0]
        for i in range(steps):
            u0 = u0 + dlt_t * (self.mu[0] * self.A @ u0 + self.mu[1] * u0 @ self.A)
            u_data.append(u0)
        return np.array(u_data)
    
    def generate_training_data(self, traj_num, steps, dlt_t = 0.001):
        u0 = np.random.rand(self.dim - 1, self.dim -1)
        u_data = self.generate_traj(steps, u0, dlt_t)
        u_x = u_data[:-1,:,:]
        u_y = u_data[1:,:,:]
        for i in range(traj_num - 1):
            u0 = np.random.rand(self.dim - 1, self.dim -1)
            u = self.generate_traj(steps, u0, dlt_t)
            u_data = np.concatenate((u_data, u), axis = 0)
            u_x = np.concatenate((u_x, u[:-1,:,:]), axis = 0)
            u_y = np.concatenate((u_y, u[1:,:,:]), axis = 0)
        u_x_lace_1 = u_x @ self.A
        u_x_lace_2 = self.A @ u_x
        self.data_x = np.reshape(u_x, (-1, 1))
        self.data_y = np.reshape(u_y, (-1, 1))
        self.data_lace_1 = np.reshape(u_x_lace_1, (-1,1))
        self.data_lace_2 = np.reshape(u_x_lace_2, (-1,1))
        # return self.data_x, self.data_y, self.data_lace_1, self.data_lace_2
    
    def testing(self, k, error_bound, acc = False):
        diff_u = self.data_y - self.data_x
        print(k)
        model_2 = k * self.data_lace_1 + self.data_lace_2
        err = 1e3
        err_history = []
        mu1_pred, y1 = Hybrid_compute_linear_weight(diff_u, self.data_lace_1)
        # err = np.linalg.norm(diff_u - y1, ord=np.inf)
        # print('k = %d, and error is %d' %(k, err))
        # err_history.append(err)
        if acc == False:
            while (err>error_bound):
                mu2_pred, y2 = Hybrid_compute_linear_weight(model_2, diff_u - y1)
                mu1_pred, y1 = Hybrid_compute_linear_weight(self.data_lace_1, diff_u - y2)
                err = np.linalg.norm(diff_u - y1 - y2, ord=np.inf)
                err_history.append(err)
                print('parameters are %f and %f, and the error is %f' %(mu1_pred, mu2_pred, err))
            return err_history
        else:
            mu2_pred, y2 = Hybrid_compute_linear_weight(model_2, diff_u - y1)
            mu1_pred, y1 = Hybrid_compute_linear_weight(self.data_lace_1, diff_u - y2)
            err = np.linalg.norm(diff_u - y1 - y2, ord=np.inf)
            err_history.append(err)
            while (err>error_bound):
                y1_pre = y1
                y2_pre = y2
                mu2_pred, y2 = Hybrid_compute_linear_weight(model_2, diff_u - y1)
                mu1_pred, y1 = Hybrid_compute_linear_weight(self.data_lace_1, diff_u - y2)
                err = np.linalg.norm(diff_u - y1 - y2, ord=np.inf)
                err_history.append(err)
                t_F = np.transpose(diff_u - y1_pre - y2_pre) @ (y1 + y2 - y1_pre - y2_pre)/(np.transpose(y1 + y2 - y1_pre - y2_pre) @ (y1 + y2 - y1_pre - y2_pre))
                y1 = t_F * y1 + (1 - t_F) * y1_pre
                print('parameters are %f and %f, the error is %f, and relaxed para is %f' %(mu1_pred, mu2_pred, err, t_F))
            return err_history
        
    def adapted_k(self):
        def objective_function(k):
            model_1 = self.data_lace_1
            model_2 = k * self.data_lace_1 + self.data_lace_2
            numerator = np.abs(np.dot(model_1.flatten(), model_2.flatten()))
            denominator = np.linalg.norm(model_1.flatten(), ord = 2) * np.linalg.norm(model_2.flatten(), ord = 2)
            return numerator / denominator

        initial_guess = 0.0
        result = minimize(objective_function, initial_guess, method='BFGS')  # 使用 BFGS 方法

        optimal_k = result.x[0]
        print(result)

        return optimal_k
                
            
            

    