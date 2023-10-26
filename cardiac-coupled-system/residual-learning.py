import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Cardiac_Electrophysiology import *
from KoopmanDL import *
from Hybrid import *


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)

# Load data
s = np.load('./data/s_sample_0416.npy')
v = np.load('./data/v_sample_0416.npy')

# Build the input tensor
DataModel = Cardiac_Electrophysiology_DataModel()
x1 = np.linspace(0,10,51)
t = np.linspace(0,10,50)
u_t = np.sin(t)
u_x_mesh = DataModel.dx(x1,x1)
u = u_t[:,np.newaxis,np.newaxis] * u_x_mesh[np.newaxis,:]

# Build v_xx
dlt_x = x1[1]-x1[0]
v_train_xx = np.zeros(np.shape(v))
v_train_xx[:,:,0] = (2 * v[:,:,1] - 2 * v[:,:,0])/dlt_x**2
for i in range(1, np.shape(v)[2]-1):
    v_train_xx[:,:,i] = (v[:,:,i-1] + v[:,:,i+1] - 2*v[:,:,i])/dlt_x**2
v_train_xx[:,:,-1] = (2 * v[:,:,-2] - 2 * v[:,:,-1])/dlt_x**2
# v_train_xx = v_train_xx[:,1:-1,1:-1]

# Build v_yy
dlt_y = x1[1]-x1[0]
v_train_yy = np.zeros(np.shape(v))
v_train_yy[:,:,0] = (2 * v[:,1,:] - 2 * v[:,0,:])/dlt_y**2
for i in range(1, np.shape(v)[1]-1):
    v_train_yy[:,i,:] = (v[:,i-1,:] + v[:,i+1,:] - 2*v[:,i,:])/dlt_y**2
v_train_yy[:,:,-1] = (2 * v[:,-2,:] - 2 * v[:,-1,:])/dlt_y**2
# v_train_yy = v_train_yy[:,1:-1,1:-1]

# Build v_data, s_data
v_train = v
s_train = s

# Build training data
v1 = np.reshape(v_train[2:,:,:],(-1, 1))
v0 = np.reshape(v_train[1:-1,:,:],(-1,1))
lacev_x0 = np.reshape(v_train_xx[1:-1,:,:],(-1,1))
lacev_y0 = np.reshape(v_train_yy[1:-1,:,:],(-1,1))

lace_data = np.concatenate((lacev_x0,lacev_y0),axis=1)

m1 = np.reshape(s_train[2:,0,:,:],(-1,1))
m0 = np.reshape(s_train[1:-1,0,:,:],(-1,1))
n1 = np.reshape(s_train[2:,1,:,:],(-1,1))
n0 = np.reshape(s_train[1:-1,1,:,:],(-1,1))
h1 = np.reshape(s_train[2:,2,:,:],(-1,1))
h0 = np.reshape(s_train[1:-1,2,:,:],(-1,1))

x_data = np.concatenate((v0,m0,n0,h0),axis=1)
y_data = np.concatenate((v1,m1,n1,h1),axis=1)
u_data = np.reshape(u[1:-1,:,:],(-1,1))

fusion_data = np.concatenate((np.reshape(x_data[:,0],(-1,1)),lace_data),axis = 1)

w, y_pred = Hybrid_compute_linear_weight(fusion_data, y_data[:,0])

model = KoopmanDL_Model(target_dim=4, u_dim=1,
                                  dic_trainable=KoopmanDL_DicNN, dic_layer_sizes=[64,64],
                                  operator_layer_sizes=[32,32])


model.Build()

# Koopman-DL pre train
lr = 0.001
opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.model_KoopmanDL.compile(optimizer=opt, loss='mse')

x_data_scaler = model.Build_x_scaler(x_data)
y_data_scaler = model.Build_y_scaler(y_data)
u_data_scaler = model.Build_u_scaler(u_data)

x_train, x_test, y_train, y_test, u_train, u_test = train_test_split(x_data_scaler, y_data_scaler, u_data_scaler, test_size=0.33, random_state=None)

tf.norm(model.model_KoopmanDL([x_train, y_train, u_train]),axis=1)

iters = 2
epochs = [2,10]
zeros_data_y_train = tf.zeros_like(model.dic.call(y_train))
zeros_data_y_test = tf.zeros_like(model.dic.call(y_test))

for i in range(iters):
    model.Train_Operator()
    model.model_KoopmanDL.fit([x_train, y_train, u_train], zeros_data_y_train, validation_data = ([x_test, y_test, u_test], zeros_data_y_test),epochs=epochs[0],verbose=0, batch_size=4096)
    model.Train_Dic()
    model.model_KoopmanDL.fit([x_train, y_train, u_train], zeros_data_y_train, validation_data = ([x_test, y_test, u_test], zeros_data_y_test),epochs=epochs[1],verbose=0, batch_size=4096)

v_linear = np.matmul(fusion_data,w)

model.Train_Operator()

err_data = np.zeros(np.shape(y_data_scaler))
err_data[:,:] = y_data[:,:]

err_data[:,0] = y_data[:,0] - np.reshape(v_linear,(1,-1))
err_data_scaler = model.Build_y_scaler(err_data)

x_train, x_test, err_train, err_test, u_train, u_test = train_test_split(x_data_scaler, err_data_scaler, u_data_scaler, test_size=0.33, random_state=None)
model.model_KoopmanDL.fit([x_train, err_train, u_train], zeros_data_y_train, validation_data = ([x_test, err_test, u_test], zeros_data_y_test),epochs=300,verbose=2, batch_size=4096)


np.save('./output/residual_x_max.npy', model.x_max)
np.save('./output/residual_x_min.npy', model.x_min)
np.save('./output/residual_y_max.npy', model.y_max)
np.save('./output/residual_y_min.npy', model.y_min)
np.save('./output/residual_u_max.npy', model.u_max)
np.save('./output/residual_u_min.npy', model.u_min)
np.save('./output/residual_w.npy', w)

model.model_KoopmanDL.save_weights('./output/residual_learning_KoopmanModel.h5')

