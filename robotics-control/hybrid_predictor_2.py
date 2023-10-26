import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import gc

import time
import numpy as np

import sys

from MPCsolver import *

if len(sys.argv) != 2:
    print("please give the external input type")
    sys.exit(1)

try:
    w_id = int(sys.argv[1])
except ValueError:
    print("Input must be a number")
    sys.exit(1)

x_data = np.load('data/x_data.npy')
y_data = np.load('data/y_data.npy')
uw_data_repeated = np.load('data/uw_data.npy')
u_data_repeated = np.load('data/u_data.npy')
w_data_repeated = np.load('data/w_data.npy')

from robotics_dym import * 
from Koopman_model import * 
from parameters import *

from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()
x_data_scaled = x_scaler.fit_transform(x_data)
y_data_scaled = x_scaler.transform(y_data)
u_scaler = MinMaxScaler()
u_data_scaled = u_scaler.fit_transform(u_data_repeated)
w_scaler = MinMaxScaler()
w_data_scaled = w_scaler.fit_transform(w_data_repeated)
uw_data_scaled = np.column_stack((u_data_scaled, w_data_scaled))


class KoopmanDL_BNN(Layer):
    """
    Koopman operator with input u
    """
    def __init__(self, n_K = 27, n_B = 2, layer_sizes=[32, 32], **kwargs):
        self.layer_sizes = layer_sizes
        self.n_K = n_K
        self.n_B = n_B
        self.input_layer = Dense(self.layer_sizes[0])
        self.hidden_layers = [Dense(layer_sizes[i], activation='tanh') for i in range(len(layer_sizes))]
        self.output_layer = Dense(self.n_K * self.n_B)
        
    def call(self, inputs_u):
        K = self.input_layer(inputs_u)
        for layer in self.hidden_layers:
            K = K + layer(K)
        outputs = self.output_layer(K)
        return tf.reshape(outputs,(-1,self.n_B,self.n_K))

Dic_3 = KoopmanDL_PsiNN()
Bnn_3 = KoopmanDL_BNN(n_K=n_K, n_B = u_dim)
input_x = Input((target_dim,))
input_y = Input((target_dim,))
input_u = Input((u_dim,))
input_w = Input((w_dim,))
psi_x = Dic_3.call(input_x)
psi_y = Dic_3.call(input_y)
model_dic_3 = Model(inputs=input_x, outputs=psi_x)
model_dic_3.trainable = False
B = Bnn_3.call(input_w)
K_layer_3 = Dense(units=n_K, use_bias = False, activation = None, trainable = False)
u_expanded = tf.expand_dims(input_u, 1)
predict = tf.matmul(u_expanded, B)
predict_reshaped = tf.reshape(predict, (-1, n_K))
outputs = predict_reshaped  + K_layer_3(psi_x) - psi_y
model_hybrid_predict = Model(inputs=[input_u, input_w], outputs=predict_reshaped)
model_hybrid_train = Model(inputs=[input_x, input_y, input_u, input_w], outputs=outputs)

input_psix = Input((n_K,))
input_u = Input((u_dim,))
u_expanded = tf.expand_dims(input_u, 1)
predict = tf.matmul(u_expanded, B)
predict_reshaped = tf.reshape(predict, (-1, n_K))
outputs = predict_reshaped  + K_layer_3(input_psix)
model_hybrid_predict_psi = Model(inputs=[input_psix, input_u, input_w], outputs=outputs)

folder_path = 'hybrid_data_2/'


if not os.path.exists(folder_path):
    os.makedirs(folder_path)

model_hybrid_train.load_weights(os.path.join(folder_path, 'model_hybrid_train_weights_Bw.h5'))
model_hybrid_predict_psi.load_weights(os.path.join(folder_path, 'model_bybrid_predict_weights_psi_Bw.h5'))


print("weight loaded from", folder_path)

load_folder_path = 'test_data/'
save_folder_path = 'hybrid_data_2/'
w_types = ['ref', 'sin', 'sinhigh', 'random']  # List of data types
w_type = w_types[w_id]

def save_data(data, name, suffix, folder_path):
    file_name = f'{name}_{suffix}.npy'
    file_path = os.path.join(folder_path, file_name)
    np.save(file_path, data)
    print(f"{file_name} saved in {folder_path} with suffix {suffix}")


def load_data(suffix, folder_path):
    s0 = np.load(os.path.join(folder_path, f's0_{suffix}.npy'))
    s_des_test = np.load(os.path.join(folder_path, f's_des_test_{suffix}.npy'))
    omega_a_ref = np.load(os.path.join(folder_path, f'omega_a_ref_{suffix}.npy'))
    omega_a_test = np.load(os.path.join(folder_path, f'omega_a_test_{suffix}.npy'))
    alpha_0_test = np.load(os.path.join(folder_path, f'alpha_0_test_{suffix}.npy'))
    alpha_a_test = np.load(os.path.join(folder_path, f'alpha_a_test_{suffix}.npy'))
    omega_a_test2 = np.load(os.path.join(folder_path, f'omega_a_test2_{suffix}.npy'))
    omega_a_test3 = np.load(os.path.join(folder_path, f'omega_a_test3_{suffix}.npy'))
    
    return s0, s_des_test, omega_a_ref, omega_a_test, alpha_0_test, alpha_a_test, omega_a_test2, omega_a_test3

suffix_set = range(10)

time_log = []
for suffix in suffix_set:
    
    s0, s_des_test, omega_a_ref, omega_a_test, alpha_0_test, alpha_a_test, omega_a_test2, omega_a_test3 = load_data(suffix, load_folder_path)

    print("Data loaded from", load_folder_path, "with suffix", suffix)

    if w_id == 0:
        omega_a = omega_a_ref
    elif w_id == 1:
        omega_a = omega_a_test
    elif w_id == 2:
        omega_a = omega_a_test2
    
    s_des = Dic_3.call(s_des_test)

    T1 = time.time()

    u_opt, s_traj = MPCsolver(s0.reshape((1,-1)), s_des, model_hybrid_predict_psi, Dic_3, omega_a.reshape((-1,1)), h, N, mpc_prob_3_scipy, x_scaler, w_scaler, u_scaler)
    T2 = time.time()

    
    print('Running Time:%s s' % (T2 - T1))
    time_log.append((T2 - T1))
    save_data(s_traj, f's_traj_{w_type}', suffix, save_folder_path)
    save_data(u_opt, f'u_opt_{w_type}', suffix, save_folder_path)

    # sys.stdout.close()
    save_data(time_log, f'time_log_{w_type}', 0, save_folder_path)

    gc.collect()

