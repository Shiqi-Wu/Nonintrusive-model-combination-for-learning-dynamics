import numpy as np
import tensorflow as tf
import KoopmanDL
from KoopmanDL import *


def Hybrid_compute_linear_weight(x,y):
    y = np.reshape(y,(-1,1))
    A = np.matmul(np.transpose(x),x)
    b = np.matmul(np.transpose(x),y)
    w = np.matmul(np.linalg.inv(A),b)
    y_pred = np.matmul(x,w)
    return w, y_pred

class Hybrid_Model(KoopmanDL.KoopmanDL_Model):
    def __init__(self, target_dim = 4, u_dim = 1, 
                 dic_trainable=KoopmanDL.KoopmanDL_DicNN, dic_layer_sizes=[64,64], n_psi_train=22,
                 operator_layer_sizes=[32, 32], **kwargs):
        # super(Hybrid_Model, self).__init__()
        self.target_dim = target_dim
        self.u_dim = u_dim
        self.dic_trainable = dic_trainable
        self.dic_layer_sizes = dic_layer_sizes
        self.n_psi_train = n_psi_train
        self.operator_layer_sizes = operator_layer_sizes
        self.K_dim = 1 + target_dim + n_psi_train
        self.loss_fun = tf.keras.losses.MeanSquaredError()
        
    def Build_linear_para(self, linear_para):
        self.linear_para = linear_para
        
    def Build_koopman_model(self, weights_loc, scaler_loc):
        self.KoopmanDLmodel = KoopmanDL.KoopmanDL_Model(self.target_dim, self.u_dim,
                                                        self.dic_trainable, self.dic_layer_sizes, self.n_psi_train,
                                                        self.operator_layer_sizes)
        self.KoopmanDLmodel.Build()
        self.KoopmanDLmodel.model_KoopmanDL.load_weights(weights_loc)
        self.KoopmanDLmodel.x_max = np.load(scaler_loc[0])
        print(self.KoopmanDLmodel.x_max)
        self.KoopmanDLmodel.x_min = np.load(scaler_loc[1])
        print(self.KoopmanDLmodel.x_min)
        self.KoopmanDLmodel.y_max = np.load(scaler_loc[2])
        self.KoopmanDLmodel.y_min = np.load(scaler_loc[3])
        self.KoopmanDLmodel.u_max = np.load(scaler_loc[4])
        self.KoopmanDLmodel.u_min = np.load(scaler_loc[5])

        
    def Compute_lace(self, u, dlt_x, dim):
        u = np.reshape(u, (dim, dim))
        u_lace1 = np.zeros(np.shape(u))
        u_lace2 = np.zeros(np.shape(u))
        u_lace1[:,0] = (2 * u[:,1] - 2 * u[:,0])/dlt_x**2
        u_lace1[:,-1] = (2 * u[:,-2] - 2 * u[:,-1])/dlt_x**2
        for i in range(1, dim - 1):
            u_lace1[:,i] = (u[:,i+1] + u[:,i-1] - 2 * u[:,i])/dlt_x**2
        u_lace2[0,:] = (2 * u[1,:] - 2 * u[0,:])/dlt_x**2
        u_lace2[-1,:] = (2 * u[-2,:] - 2 * u[-1,:])/dlt_x**2
        for i in range(1, dim - 1):
            u_lace2[i,:] = (u[i-1,:] + u[i+1,:] - 2 * u[i,:])/dlt_x**2
        u_lace1 = np.reshape(u_lace1, (-1, 1))
        u_lace2 = np.reshape(u_lace2, (-1, 1))
        return np.concatenate((u_lace1, u_lace2), axis = 1)
    
    def Compute_linear_predict(self, v0, dlt_x, dim):
        v_lace = self.Compute_lace(v0, dlt_x, dim)
        v0 = np.reshape(v0, (-1,1))
        v0_data = np.concatenate((v0, v_lace), axis = 1)
        v1 = np.matmul(v0_data, self.linear_para)
        return v1
    
    def Long_time_predict(self, u0, steps, dlt_x, dim, inputs):
        predict_results = [u0]
        inputs_scaler = self.KoopmanDLmodel.transform_u(inputs)
        for step in range(1, steps + 1):
            u0_scaler = self.KoopmanDLmodel.transform_x(u0)
            u1_scaler = self.KoopmanDLmodel.Predict(u0_scaler, inputs_scaler[step - 1,:])
            u1 = self.KoopmanDLmodel.inverse_transform_y(u1_scaler)
            u1 = tf.convert_to_tensor(u1)
            # u1[:, 0] = u1[:, 0] + tf.squeeze(self.Compute_linear_predict(u0[:, 0], dlt_x, dim))
            u1 = tf.concat([tf.expand_dims(u1[:, 0] + tf.squeeze(self.Compute_linear_predict(u0[:, 0], dlt_x, dim)), axis=1), u1[:, 1:]], axis=1)
            predict_results.append(u1)
            u0 = u1
        return predict_results

class Intrusive_combinde_model(object):
    def __init__(self, target1_dim = 3, target2_dim = 4, u_dim = 1, 
                 dic_trainable=KoopmanDL_DicNN, dic_layer_sizes=[64,64], n_psi_train=22,
                 operator_layer_sizes=[32, 32], **kwargs):
        self.target1_dim = target1_dim
        self.target2_dim = target2_dim
        self.u_dim = u_dim
        self.dic_trainable = dic_trainable
        self.dic_layer_sizes = dic_layer_sizes
        self.n_psi_train = n_psi_train
        self.operator_layer_sizes = operator_layer_sizes
        self.K_dim = 1 + target2_dim + n_psi_train
        self.loss_fun = tf.keras.losses.MeanSquaredError()
        super(Intrusive_combinde_model, self).__init__()
    
    def Build(self):
        self.mu_layer = Dense(1, use_bias=False, dtype=tf.float64)
        self.dic = KoopmanDL_PsiNN(self.dic_trainable,
                              self.dic_layer_sizes,
                              self.n_psi_train)
        self.knn = KoopmanDL_KNN(self.K_dim,
                            self.operator_layer_sizes)
        
        inputs_x_laplace = Input((self.target1_dim,))
        inputs_x = Input((self.target2_dim,))
        inputs_u = Input((self.u_dim,))
        
        self.model_psi = Model(inputs=inputs_x, outputs=self.dic.call(inputs_x))
        psi_x = self.model_psi(inputs_x)
        
        self.model_k = Model(inputs=inputs_u,outputs=self.knn.call(inputs_u))
        psi_x = tf.expand_dims(psi_x, 1)
        outputs = tf.matmul(psi_x, self.model_k(inputs_u))
        outputs = tf.reshape(outputs,(tf.shape(psi_x)[0],-1))

        modification = self.mu_layer(inputs_x_laplace)
        outputs = outputs[:, 1:self.target2_dim+1] + tf.concat([modification, tf.zeros_like(outputs[:, 1:self.target2_dim+1])[:, 1:]], axis=1)
        self.combined_model = Model(inputs=[inputs_x, inputs_x_laplace, inputs_u], outputs=outputs)
        return

    def scale(self, x_data, y_data, x_laplace_data, u_data):
        self.scalerx = MinMaxScaler()
        self.scalerlaplace = MinMaxScaler()
        self.scaleru = MinMaxScaler()
        x_data_scaled = self.scalerx.fit_transform(x_data)
        y_data_scaled = self.scalerx.transform(y_data)
        u_data_scaled = self.scaleru.fit_transform(u_data)
        x_laplace_data_scaled = self.scalerlaplace.fit_transform(x_laplace_data)
        return x_data_scaled, y_data_scaled, x_laplace_data_scaled, u_data_scaled
    
    def compute_laplace(self, u, dlt_spatial):
        dim_square = np.shape(u)[0]
        dim = int(np.sqrt(dim_square))
        u = np.reshape(u, (dim, dim))
        u_lace1 = np.zeros(np.shape(u))
        u_lace2 = np.zeros(np.shape(u))
        u_lace1[:,0] = (2 * u[:,1] - 2 * u[:,0])/dlt_spatial**2
        u_lace1[:,-1] = (2 * u[:,-2] - 2 * u[:,-1])/dlt_spatial**2
        for i in range(1, dim - 1):
            u_lace1[:,i] = (u[:,i+1] + u[:,i-1] - 2 * u[:,i])/dlt_spatial**2
        u_lace2[0,:] = (2 * u[1,:] - 2 * u[0,:])/dlt_spatial**2
        u_lace2[-1,:] = (2 * u[-2,:] - 2 * u[-1,:])/dlt_spatial**2
        for i in range(1, dim - 1):
            u_lace2[i,:] = (u[i-1,:] + u[i+1,:] - 2 * u[i,:])/dlt_spatial**2
        u_lace1 = np.reshape(u_lace1, (-1, 1))
        u_lace2 = np.reshape(u_lace2, (-1, 1))
        return np.concatenate((u.reshape((-1,1)), u_lace1, u_lace2), axis = 1)

    def long_time_predict(self, x0, steps, inputs, dlt_spatial):
        predict_results = [x0]
        x0_scaled = self.scalerx.transform(x0)
        for step in range(1, steps + 1):
            inputs_current = inputs[step - 1,:].reshape((-1, 1))
            inputs_current_scaled = self.scaleru.transform(inputs_current)
            x0_laplace = self.compute_laplace(x0[:,0], dlt_spatial)
            x0_laplace_scaled = self.scalerlaplace.transform(x0_laplace)
            x1_scaled = self.combined_model([x0_scaled, x0_laplace_scaled, inputs_current_scaled])
            x1 = self.scalerx.inverse_transform(x1_scaled)
            predict_results.append(x1)
            x0_scaled = x1_scaled
        return predict_results
    