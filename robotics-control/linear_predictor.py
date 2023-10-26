import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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


Dic_1 = KoopmanDL_PsiNN()
input_x = Input((target_dim,))
input_y = Input((target_dim,))
input_uw = Input((u_dim + w_dim,))
psi_x = Dic_1.call(input_x)
psi_y = Dic_1.call(input_y)
model_dic_1 = Model(inputs=input_x, outputs=psi_x)
K_layer = Dense(units=1 + target_dim + Dic_1.n_psi_train, use_bias = False, activation = None, trainable = False)
B_layer = Dense(units=1 + target_dim + Dic_1.n_psi_train, use_bias = False, activation = None, trainable = False)
predict = K_layer(psi_x) + B_layer(input_uw)
outputs = predict - psi_y
model_linear_train = Model(inputs = [input_x, input_y, input_uw], outputs = outputs)
model_linear_predict = Model(inputs = [input_x, input_uw], outputs = predict)

input_psix = Input((n_K,))
psi_predict = K_layer(input_psix) + B_layer(input_uw)
model_linear_predict_psi = Model(inputs = [input_psix, input_uw], outputs = psi_predict)

import numpy as np
import os

folder_path = 'data/'

model_dic_1.load_weights(os.path.join(folder_path, 'dictionary_weights.h5'))


print("weight load from", folder_path)

psi_x = Dic_1.call(x_data_scaled)
psi_y = Dic_1.call(y_data_scaled)

combine_data = tf.concat([psi_x, uw_data_scaled], axis=1)
XT_Y = tf.matmul(tf.transpose(combine_data), psi_y)
XT_X_inv = tf.linalg.inv(tf.matmul(tf.transpose(combine_data), combine_data))
result = tf.matmul(XT_X_inv, XT_Y)
K_layer_weights = K_layer.get_weights()
B_layer_weights = B_layer.get_weights()
K_layer_weights[0] = result[:n_K, :]
B_layer_weights[0] = result[n_K:, :]
K_layer.set_weights(K_layer_weights)
B_layer.set_weights(B_layer_weights)

load_folder_path = 'test_data/'
save_folder_path = 'linear_data/'
w_types = ['ref', 'sin', 'random']  # List of data types
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
    
    s_des = Dic_1.call(s_des_test)

    T1 = time.time()
    u_opt, s_traj = MPCsolver(s0.reshape((1,-1)), s_des, model_linear_predict_psi, Dic_1, omega_a.reshape((-1,1)), h, N, mpc_prob_1_scipy, x_scaler, w_scaler, u_scaler)
    T2 = time.time()

    
    print('Running Time:%s s' % (T2 - T1))
    time_log.append((T2 - T1))
    save_data(s_traj, f's_traj_{w_type}', suffix, save_folder_path)
    save_data(u_opt, f'u_opt_{w_type}', suffix, save_folder_path)

    # sys.stdout.close()
    save_data(time_log, f'time_log_{w_type}', 0, save_folder_path)

    gc.collect()