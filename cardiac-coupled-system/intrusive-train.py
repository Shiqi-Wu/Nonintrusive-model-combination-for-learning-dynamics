import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
# from sklearn import preprocessing
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
s = np.load('./data/s_sample.npy')
v = np.load('./data/v_sample.npy')

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

# Build v_yy
dlt_y = x1[1]-x1[0]
v_train_yy = np.zeros(np.shape(v))
v_train_yy[:,:,0] = (2 * v[:,1,:] - 2 * v[:,0,:])/dlt_y**2
for i in range(1, np.shape(v)[1]-1):
    v_train_yy[:,i,:] = (v[:,i-1,:] + v[:,i+1,:] - 2*v[:,i,:])/dlt_y**2
v_train_yy[:,:,-1] = (2 * v[:,-2,:] - 2 * v[:,-1,:])/dlt_y**2

# Build v_data, s_data
v_train = v
s_train = s

# Build training data
v1 = np.reshape(v_train[2:,:,:],(-1, 1))
v0 = np.reshape(v_train[1:-1,:,:],(-1,1))
laplacev_x0 = np.reshape(v_train_xx[1:-1,:,:],(-1,1))
laplacev_y0 = np.reshape(v_train_yy[1:-1,:,:],(-1,1))

laplace_data = np.concatenate((laplacev_x0, laplacev_y0),axis=1)

m1 = np.reshape(s_train[2:,0,:,:],(-1,1))
m0 = np.reshape(s_train[1:-1,0,:,:],(-1,1))
n1 = np.reshape(s_train[2:,1,:,:],(-1,1))
n0 = np.reshape(s_train[1:-1,1,:,:],(-1,1))
h1 = np.reshape(s_train[2:,2,:,:],(-1,1))
h0 = np.reshape(s_train[1:-1,2,:,:],(-1,1))

x_data = np.concatenate((v0,m0,n0,h0),axis=1)
y_data = np.concatenate((v1,m1,n1,h1),axis=1)
u_data = np.reshape(u[1:-1,:,:],(-1,1))

fusion_data = np.concatenate((np.reshape(x_data[:,0],(-1,1)),laplace_data),axis = 1)


model =Intrusive_combinde_model(target1_dim = 3, target2_dim = 4, u_dim=1,
                                  dic_trainable=KoopmanDL_DicNN, dic_layer_sizes=[64,64],
                                  operator_layer_sizes=[32,32])

model.Build()

# Training
lr = 0.0001
opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.combined_model.compile(optimizer=opt, loss='mse')

x_data_scaled, y_data_scaled, fusion_data_scaled, u_data_scaled = model.scale(x_data, y_data, fusion_data, u_data)
x_train, x_test, y_train, y_test, fusion_train, fusion_test, u_train, u_test = train_test_split(x_data_scaled, y_data_scaled, fusion_data_scaled, u_data_scaled, test_size=0.33, random_state=None)

history = model.combined_model.fit([x_train, fusion_train, u_train], y_train, validation_data = ([x_test, fusion_test, u_test], y_test),epochs=3000,verbose=2, batch_size=4096)


history_data = history.history

file_path = 'intrusive_training_history.npz'
np.savez(file_path, **history_data)

model.combined_model.save_weights('intrusive_combined_model.h5')


print("The intrusive training of combined model is done.")
    
    
