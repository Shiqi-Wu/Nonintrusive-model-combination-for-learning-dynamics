import numpy as np
import tensorflow as tf
import KoopmanDL


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
            
    