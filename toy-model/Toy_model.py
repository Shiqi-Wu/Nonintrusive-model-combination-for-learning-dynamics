import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import tensorflow as tf
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Toymodel(object):
    def __init__(self, dim = 20, L0 = 0, L1 = 1, mu = 1):
        self.dim = dim
        self.L0 = L0
        self.L1 = L1
        self.mu = mu
        self.build_A()
    
    def reaction_term(self, u):
        return 1/4*2 * (u-1)*(u+1)**2+ 1/4 * 2 * (u-1)**2 * (u+1)
    
    def build_A(self):
        self.A = np.zeros([self.dim - 1, self.dim - 1])
        for i in range(self.dim - 1):
            self.A[i, i] = -2 
        for i in range(self.dim - 2):
            self.A[i, i + 1] = 1
            self.A[i + 1, i] = 1

    def ivp_function(self, t, u):
        dlt_x = (self.L1 - self.L0)/self.dim
        du = self.A @ u / dlt_x ** 2 + np.array([self.reaction_term(ui) for ui in u])
        return du
    
    def generate_traj(self, steps, u0, dlt_t = 0.001):
        u_data = np.zeros([steps + 1, self.dim - 1])
        u_data[0, :] = u0
        dlt_x = (self.L1 - self.L0)/self.dim
        for step in range(steps):
            u_0 = u_data[step, :]
            u_1 = u_0 + dlt_t * (self.mu * u_0 @ self.A/ dlt_x ** 2 + self.reaction_term(u_0))
            u_data[step + 1,:] = u_1
        return u_data
    
    def generate_traj_solve_ivp(self, steps, u0, dlt_t = 0.001):
        t_span = [0, steps * dlt_t]
        u_data = np.zeros([steps + 1, self.dim - 1])
        u_data[0, :] = u0
        dlt_x = (self.L1 - self.L0)/self.dim

        def ivp_function(t, u):
            return self.mu * u @ self.A / dlt_x ** 2 + self.reaction_term(u)

        sol = solve_ivp(ivp_function, t_span, u0, t_eval=np.linspace(t_span[0], t_span[1], steps + 1))
        u_data = sol.y.T
        print(np.shape(u_data))
        return u_data
    
    def genarate_training_data(self, steps, traj_num, dlt_t = 0.001, type = 'Euler-forward'):
        u0 = 2 * np.random.rand(self.dim - 1) - 1

        if type == 'Euler-forward':
            u_data = self.generate_traj(steps, u0, dlt_t)
        elif type == 'Solve_ivp':
            u_data = self.generate_traj_solve_ivp(steps, u0, dlt_t)

        u_x = u_data[:-1,:]
        u_y = u_data[1:,:]
        for i in range(traj_num - 1):
            u0 = np.random.rand(self.dim - 1)
            if type == 'Euler-forward':
                u = self.generate_traj(steps, u0, dlt_t)
            elif type == 'Solve_ivp':
                u = self.generate_traj_solve_ivp(steps, u0, dlt_t)
            u_data = np.concatenate((u_data, u), axis = 0)
            u_x = np.concatenate((u_x, u[:-1,:]), axis = 0)
            u_y = np.concatenate((u_y, u[1:,:]), axis = 0)
        u_x_lace = u_x @ self.A
        u_y_lace = u_y @ self.A
        data_x = np.reshape(u_x, (-1, 1))
        data_y = np.reshape(u_y, (-1, 1))
        data_lace_x = np.reshape(u_x_lace, (-1,1))
        data_lace_y = np.reshape(u_y_lace, (-1,1))
        return data_x, data_y, data_lace_x, data_lace_y
    
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

def Original_Hybrid_Method_residual(x_data, y_data, x_lace, dic, mu_error_bound, dlt_t):
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
        y_koopman = y_data - x_data - y_linear_pred
        y_koopman_rescale = y_koopman / dlt_t
        dic_y_koopman = dic.call(y_koopman_rescale)
        K = inverse_matrix @ dic_x.T @ dic_y_koopman
        dic_y_koopman_pred = dic_x @ K
        y_koopman_pred_rescale = np.reshape(dic_y_koopman_pred[:,1],(-1,1))
        y_koopman_pred = y_koopman_pred_rescale * dlt_t
        y_linear = y_data - x_data - y_koopman_pred
        mu_history.append(mu_pred)
        err_history.append(np.linalg.norm(y_linear - y_linear_pred, ord=np.inf))
        koopman_history.append(y_koopman_pred_rescale)
        print('epoch = %d, the error is %f' %(epoch, err_history[-1]))
        epoch = epoch + 1
    
    return mu_pred, K, mu_history, err_history, koopman_history

# Redefine the DicPoly class for TensorFlow
class DicPolyTF(object):
    def __init__(self, degree=2):
        self.dictionary_degree = degree
    
    def call(self, data):
        poly = []
        poly_index = []
        pre_poly = []
        N = data.shape[1]
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
            cur_poly_data = tf.ones(data.shape[0])
            for i in index:
                cur_poly_data = cur_poly_data * data[:, i]
            poly.append(cur_poly_data)
        poly = tf.stack(poly, axis=1)
        ones = tf.ones((data.shape[0], 1))
        return tf.concat([ones, data, poly], axis=1)

# Define the function for gradient descent optimization using TensorFlow
def optimize_with_gradient_descent_tf(x_data, y_data, lace_data, degree, learning_rate, iterations):
    dic_poly_tf = DicPolyTF(degree)

    # Convert NumPy arrays to TensorFlow tensors
    x_tf = tf.convert_to_tensor(dic_poly_tf.call(x_data), dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y_data, dtype=tf.float32)
    lace_tf = tf.convert_to_tensor(lace_data, dtype=tf.float32)

    # Initialize parameters
    mu = tf.Variable(0.1, dtype=tf.float32)
    lam = tf.Variable(tf.random.uniform([x_tf.shape[1]], dtype=tf.float32))

    # Initialize error history list
    err_history = []

    # Perform gradient descent
    for i in range(iterations):
        with tf.GradientTape() as tape:
            # Compute predicted values
            y_pred = mu * lace_tf + tf.linalg.matvec(x_tf, lam)
            
            # Calculate the loss
            loss = tf.reduce_mean(tf.square(y_pred - y_tf))

        # Compute gradients
        gradients = tape.gradient(loss, [mu, lam])

        # Update parameters
        mu.assign_sub(learning_rate * gradients[0])
        lam.assign_sub(learning_rate * gradients[1])

        # Append error (infinite norm) to the history
        err = tf.norm(y_pred - y_tf, ord=np.inf).numpy()
        err_history.append(err)

        print(f"Iteration {i+1}, Loss: {loss.numpy()}")

    # Return the final parameters
    return mu.numpy(), lam.numpy(), err_history

def optimize_with_gradient_descent_tf_small_batch(x_data, y_data, lace_data, degree, learning_rate, iterations, batch_size):
    dic_poly_tf = DicPolyTF(degree)
    x_tf_full = tf.convert_to_tensor(dic_poly_tf.call(x_data), dtype=tf.float32)
    y_tf_full = tf.convert_to_tensor(y_data, dtype=tf.float32)
    lace_tf_full = tf.convert_to_tensor(lace_data, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((x_tf_full, y_tf_full, lace_tf_full))
    dataset = dataset.batch(batch_size)

    # Initialize parameters
    mu = tf.Variable(0.1, dtype=tf.float32)
    lam = tf.random.uniform([x_tf_full.shape[1]], dtype=tf.float32)
    lam = tf.Variable(tf.reshape(lam, [-1, 1]))

    # Initialize error history list
    err_history = []

    # Perform gradient descent
    for i in range(iterations):
        for x_batch, y_batch, lace_batch in dataset:
            with tf.GradientTape() as tape:
                # print(tf.shape(lam))
                y_pred = mu * lace_batch + tf.matmul(x_batch, lam)
                # print(tf.shape(mu * lace_batch))
                # print(tf.shape(tf.matmul(x_batch, lam)))
                # print(tf.shape(y_pred))
                # print(tf.shape(y_batch))
                loss = tf.reduce_mean(tf.square(y_pred - y_batch))
            
            # break
            gradients = tape.gradient(loss, [mu, lam])
            mu.assign_sub(learning_rate * gradients[0])
            lam.assign_sub(learning_rate * gradients[1])
            tf.keras.backend.clear_session()
            gc.collect()

        # break
        # Compute error for the entire dataset (for monitoring)
        y_pred_full = mu * lace_tf_full + tf.matmul(x_tf_full, lam)
        err = tf.norm(y_pred_full - y_tf_full, ord=np.inf).numpy()
        err_history.append(err)

        print(f"Iteration {i+1}, Loss: {loss.numpy()}")

    # Return the final parameters
    return mu.numpy(), lam.numpy(), err_history


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
        result = minimize(objective_function, initial_guess, method='L-BFGS-B')

        optimal_k = result.x[0]
        print(result)

        return optimal_k
                
class LinearModelCombiningNN(object):
    def __init__(self):
        self.mu = 0.1
        self.NN = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.scaler = MinMaxScaler() 

    def train_NN(self, x_data, y_data, epochs=100, learning_rate=0.0001):
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        y_data_scaled = self.scaler.fit_transform(y_data)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_scaled, test_size=0.2, random_state=42)

        self.NN.compile(optimizer=optimizer, loss=loss_fn)
        self.NN.fit(x_train, y_train, epochs=epochs, validation_data = [x_test, y_test], verbose = 2)

    def train_NN_central_difference(self, x1_data, x2_data, y_data, epochs=100, learning_rate=0.0001):
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        y_data_scaled = self.scaler.fit_transform(y_data)
        x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_data, x2_data, y_data_scaled, test_size=0.2, random_state=42)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # Forward pass: Compute predictions by running inputs through the model
                y_pred_train = (self.NN(x1_train) + self.NN(x2_train)) / 2
                # Compute training loss
                loss_train = loss_fn(y_train, y_pred_train)

            # Compute gradients
            gradients = tape.gradient(loss_train, self.NN.trainable_variables)
            # Update weights
            optimizer.apply_gradients(zip(gradients, self.NN.trainable_variables))

            # Optional: Compute validation loss for monitoring
            y_pred_test = (self.NN(x1_test) + self.NN(x2_test)) / 2
            loss_test = loss_fn(y_test, y_pred_test)

            # Print epoch, training and validation loss
            print(f"Epoch {epoch + 1}, Training Loss: {loss_train.numpy()}, Validation Loss: {loss_test.numpy()}")

    def NN_predict(self, x):
        NN_predict_scaled = self.NN(x).numpy()
        NN_predict = self.scaler.inverse_transform(NN_predict_scaled)
        return NN_predict
    
    def NN_predict_central_difference(self, x1, x2):
        NN_predict_scaled = (self.NN(x1) + self.NN(x2)) / 2
        NN_predict = self.scaler.inverse_transform(NN_predict_scaled)
        return NN_predict

    def linear_regression(self, x, y):
        x = np.array(x)
        y = np.array(y)

        # Calculate the mean of x and y
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Calculate the numerator and denominator for the linear regression formula
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        # Update self.mu
        self.mu = numerator / denominator

    def iterative_train(self, x_data, y_data, 
                        x_lace_data, y_lace_data, 
                        iterative_steps = 50, epochs = 100, type = 'Euler-forward', 
                        acc = False):
        dx_data = y_data - x_data
        mu_pre = 1e3
        err_history = []

        if acc == True:
            if type == 'Euler-forward':
                self.linear_regression(x_lace_data, dx_data)
                mu_pre = self.mu
                dx_data_NN = dx_data - self.mu * x_lace_data
                self.train_NN(x_data, dx_data_NN, epochs)
                dx_data_NN_predict = self.NN_predict(x_data)
                dx_data_linear = dx_data - dx_data_NN_predict
                self.linear_regression(x_lace_data, dx_data_linear)
                dx_data_predict_previous = self.mu * x_lace_data + dx_data_NN_predict

                err = tf.norm(dx_data_linear - self.mu * x_lace_data, ord=np.inf).numpy()
                err_history.append(err)

                for step in range(iterative_steps):
                    mu_pre = self.mu
                    print(self.mu)
                    dx_data_NN = dx_data - self.mu * x_lace_data
                    self.train_NN(x_data, dx_data_NN, epochs)
                    dx_data_NN_predict = self.NN_predict(x_data)
                    dx_data_linear = dx_data - dx_data_NN_predict
                    self.linear_regression(x_lace_data, dx_data_linear)
                    dx_data_predict_current = self.mu * x_lace_data + dx_data_NN_predict

                    numerator = tf.reduce_sum((dx_data - dx_data_predict_previous) * (dx_data_predict_previous - dx_data_predict_current))
                    denominator = tf.reduce_sum(tf.square(dx_data_predict_previous - dx_data_predict_current))
                    t_F = numerator / denominator

                    self.mu = t_F * self.mu + (1 - t_F) * mu_pre

                    dx_data_predict_previous = dx_data_predict_current

                    err = tf.norm(dx_data_linear - self.mu * x_lace_data, ord=np.inf).numpy()
                    err_history.append(err)
                
            elif type == 'Euler-backward':
                self.linear_regression(y_lace_data, dx_data)
                mu_pre = self.mu

                dx_data_NN = dx_data - self.mu * x_lace_data
                self.train_NN(y_data, dx_data_NN, epochs)
                dx_data_NN_predict = self.NN_predict(y_data)
                dx_data_linear = dx_data - dx_data_NN_predict
                self.linear_regression(y_lace_data, dx_data_linear)
                dx_data_predict_previous = self.mu * y_lace_data + dx_data_NN_predict

                err = tf.norm(dx_data_linear - self.mu * y_lace_data, ord=np.inf).numpy()
                err_history.append(err)

                for step in range(iterative_steps):
                    mu_pre = self.mu
                    dx_data_NN = dx_data - self.mu * y_lace_data
                    self.train_NN(y_data, dx_data_NN, epochs)
                    dx_data_NN_predict = self.NN_predict(y_data)
                    dx_data_linear = dx_data - dx_data_NN_predict
                    self.linear_regression(y_lace_data, dx_data_linear)
                    dx_data_predict_current = self.mu * y_lace_data + dx_data_NN_predict

                    numerator = tf.reduce_sum((dx_data - dx_data_predict_previous) * (dx_data_predict_previous - dx_data_predict_current))
                    denominator = tf.reduce_sum(tf.square(dx_data_predict_previous - dx_data_predict_current))
                    t_F = numerator / denominator

                    self.mu = t_F * self.mu + (1 - t_F) * mu_pre

                    dx_data_predict_previous = dx_data_predict_current

                    err = tf.norm(dx_data_linear - self.mu * y_lace_data, ord=np.inf).numpy()
                    err_history.append(err)

            elif type == 'Central-difference':
                self.linear_regression((x_lace_data + y_lace_data)/2, dx_data)
                mu_pre = self.mu

                dx_data_NN = dx_data - self.mu * (x_lace_data + y_lace_data)/2
                self.train_NN_central_difference(x_data, y_data, dx_data_NN, epochs)
                dx_data_NN_predict = self.NN_predict_central_difference(x_data, y_data)
                dx_data_linear = dx_data - dx_data_NN_predict
                self.linear_regression((x_lace_data + y_lace_data)/2, dx_data_linear)
                dx_data_predict_previous = self.mu * (x_lace_data + y_lace_data)/2 + dx_data_NN_predict

                err = tf.norm(dx_data_linear - self.mu * y_lace_data, ord=np.inf).numpy()
                err_history.append(err)

                for step in range(iterative_steps):
                    mu_pre = self.mu
                    dx_data_NN = dx_data - self.mu * (x_lace_data + y_lace_data)/2
                    self.train_NN_central_difference(x_data, y_data, dx_data_NN, epochs)
                    dx_data_NN_predict = self.NN_predict_central_difference(x_data, y_data)
                    dx_data_linear = dx_data - dx_data_NN_predict
                    self.linear_regression((x_lace_data + y_lace_data)/2, dx_data_linear)
                    dx_data_predict_current = self.mu * (x_lace_data + y_lace_data)/2 + dx_data_NN_predict

                    numerator = tf.reduce_sum((dx_data - dx_data_predict_previous) * (dx_data_predict_previous - dx_data_predict_current))
                    denominator = tf.reduce_sum(tf.square(dx_data_predict_previous - dx_data_predict_current))
                    t_F = numerator / denominator

                    self.mu = t_F * self.mu + (1 - t_F) * mu_pre

                    dx_data_predict_previous = dx_data_predict_current

                    err = tf.norm(dx_data_linear - self.mu * (x_lace_data + y_lace_data)/2, ord=np.inf).numpy()
                    err_history.append(err)


        elif acc == False:
            if type == 'Euler-forward':
                self.linear_regression(x_lace_data, dx_data)
                for step in range(iterative_steps):
                    mu_pre = self.mu
                    print(self.mu)
                    dx_data_NN = dx_data - self.mu * x_lace_data
                    self.train_NN(x_data, dx_data_NN, epochs)
                    dx_data_NN_predict = self.NN_predict(x_data)
                    dx_data_linear = dx_data - dx_data_NN_predict
                    self.linear_regression(x_lace_data, dx_data_linear)
                    err = tf.norm(dx_data_linear - self.mu * x_lace_data, ord=np.inf).numpy()
                    err_history.append(err)

            elif type == 'Euler-backward':
                self.linear_regression(y_lace_data, dx_data)
                for step in range(iterative_steps):
                    mu_pre = self.mu
                    dx_data_NN = dx_data - self.mu * y_lace_data
                    self.train_NN(y_data, dx_data_NN, epochs)
                    dx_data_NN_predict = self.NN_predict(y_data)
                    dx_data_linear = dx_data - dx_data_NN_predict
                    self.linear_regression(y_lace_data, dx_data_linear)
                    err = tf.norm(dx_data_linear - self.mu * y_lace_data, ord=np.inf).numpy()
                    err_history.append(err)
        
            elif type == 'Central-difference':
                self.linear_regression((x_lace_data + y_lace_data)/2, dx_data)
                for step in range(iterative_steps):
                    mu_pre = self.mu
                    dx_data_NN = dx_data - self.mu * (x_lace_data + y_lace_data)/2
                    self.train_NN_central_difference(x_data, y_data, dx_data_NN, epochs)
                    dx_data_NN_predict = self.NN_predict_central_difference(x_data, y_data)
                    dx_data_linear = dx_data - dx_data_NN_predict
                    self.linear_regression((x_lace_data + y_lace_data)/2, dx_data_linear)
                    err = tf.norm(dx_data_linear - self.mu * (x_lace_data + y_lace_data)/2, ord=np.inf).numpy()
                    err_history.append(err)

        return err_history
        
        



            

    