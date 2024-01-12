from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.layers import Input, Add, Multiply, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from sklearn.preprocessing import MinMaxScaler
tf.keras.backend.set_floatx('float64')

class KoopmanDL_DicNN(Layer):
    """
    Trainable disctionries
    """
    
    def __init__(self, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        """_summary_

        Args:
            layer_sizes (list, optional): Number of unit of hidden layer, activation = 'tanh'. Defaults to [64, 64].
            n_psi_train (int, optional): Number of unit of output layer. Defaults to 22.
        """
        super(KoopmanDL_DicNN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.input_layer = Dense(self.layer_sizes[0], name='Dic_input', use_bias=False)
        self.hidden_layers = [Dense(layer_sizes[i], activation='tanh', name='Dic_hidden_%d'%i) for i in range(len(layer_sizes))]        
        self.output_layer = Dense(n_psi_train, name='Dic_output')
        self.n_psi_train = n_psi_train
        
    def call(self, inputs):
        psi_x_train = self.input_layer(inputs)
        for layer in self.hidden_layers:
            psi_x_train = psi_x_train + layer(psi_x_train)
        outputs = self.output_layer(psi_x_train)
        return outputs
    
    def get_config(self):
        config = super(KoopmanDL_DicNN, self).get_config()
        config.update({
            'layer_sizes': self.layer_sizes,
            'n_psi_train': self.n_psi_train
        })
        return config

class KoopmanDL_PsiNN(Layer):
    """Concatenate constant, data and trainable dictionaries together as [1, data, DicNN]

    """
    
    def __init__(
        self,
        dic_trainable=KoopmanDL_DicNN,
        layer_sizes=[64,64],
        n_psi_train=22,
        **kwargs):
        super(KoopmanDL_PsiNN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.dic_trainable = dic_trainable
        self.n_dic_customized = n_psi_train
        self.dicNN = self.dic_trainable(
            layer_sizes=self.layer_sizes,
            n_psi_train=self.n_dic_customized)
    
    def call(self, inputs):
        constant = tf.ones_like(tf.slice(inputs, [0, 0], [-1, 1]))
        psi_x_train = self.dicNN(inputs)
        outputs = Concatenate()([constant, inputs, psi_x_train])
        return outputs
    
    def generate_B(self, inputs):
        target_dim = inputs.shape[-1]
        self.basis_func_number = self.n_dic_customized + target_dim + 1
        # Form B matrix
        self.B = np.zeros((self.basis_func_number, target_dim))
        for i in range(0, target_dim):
            self.B[i + 1][i] = 1
        return self.B

    def get_config(self):
        config = super(KoopmanDL_PsiNN, self).get_config()
        config.update({
            'dic_trainable': self.dic_trainable,
            'layer_sizes': self.layer_sizes,
            'n_psi_train': self.n_dic_customized
        })
        return config
    
class KoopmanDL_KNN(Layer):
    """
    Koopman operator with input u
    """
    def __init__(self, n_K=27, layer_sizes=[32, 32], **kwargs):
        self.layer_sizes = layer_sizes
        self.n_K = n_K
        self.input_layer = Dense(self.layer_sizes[0], name='K_input')
        self.hidden_layers = [Dense(layer_sizes[i], activation='tanh', name='K_hidden_%d'%i) for i in range(len(layer_sizes))]
        self.output_layer = Dense(self.n_K**2, name='K_output')
        
    def call(self, inputs_u):
        K = self.input_layer(inputs_u)
        for layer in self.hidden_layers:
            K = K + layer(K)
        outputs = self.output_layer(K)
        return tf.reshape(outputs,(-1,self.n_K,self.n_K))
    
class KoopmanDL_Model(KoopmanDL_PsiNN, KoopmanDL_KNN):
    """
    Koopman with input
    """
    def __init__(self, target_dim = 4, u_dim = 1, 
                 dic_trainable=KoopmanDL_DicNN, dic_layer_sizes=[64,64], n_psi_train=22,
                 operator_layer_sizes=[32, 32], **kwargs):
        self.target_dim = target_dim
        self.u_dim = u_dim
        self.dic_trainable = dic_trainable
        self.dic_layer_sizes = dic_layer_sizes
        self.n_psi_train = n_psi_train
        self.operator_layer_sizes = operator_layer_sizes
        self.K_dim = 1 + target_dim + n_psi_train
        self.loss_fun = tf.keras.losses.MeanSquaredError()
        super(KoopmanDL_Model, self).__init__()
    
    def Build_x_scaler(self, x_data):
        self.x_max = np.max(x_data, axis = 0)
        self.x_min = np.min(x_data, axis = 0)
        return (x_data - self.x_min)/(self.x_max - self.x_min)
    
    def Build_y_scaler(self, y_data):
        self.y_max = np.max(y_data, axis = 0)
        self.y_min = np.min(y_data, axis = 0)
        return (y_data - self.y_min)/(self.y_max - self.y_min)
    
    def Build_u_scaler(self, u_data):
        self.u_max = np.max(u_data, axis = 0)
        self.u_min = np.min(u_data, axis = 0)
        return (u_data - self.u_min)/(self.u_max - self.u_min)
    
    def transform_x(self, x_data):
        return (x_data - self.x_min)/(self.x_max - self.x_min)
    
    def transform_y(self, y_data):
        return (y_data - self.y_min)/(self.y_max - self.y_min)

    def transform_u(self, u_data):
        return (u_data - self.u_min)/(self.u_max - self.u_min)
    
    def inverse_transform(self, data_scaler, data_max, data_min):
        return data_scaler * (data_max - data_min) + data_min
    
    def inverse_transform_x(self, x_data_scaler):
        return self.inverse_transform(x_data_scaler, self.x_max, self.x_min)
    
    def inverse_transform_y(self, y_data_scaler):
        return self.inverse_transform(y_data_scaler, self.y_max, self.y_min)
    
    def inverse_transform_u(self, u_data_scaler):
        return self.inverse_transform(u_data_scaler, self.u_max, self.u_min)
    
    def Build(self):
        self.dic = KoopmanDL_PsiNN(self.dic_trainable,
                              self.dic_layer_sizes,
                              self.n_psi_train)
        self.knn = KoopmanDL_KNN(self.K_dim,
                            self.operator_layer_sizes)
        inputs_x = Input((self.target_dim,))
        inputs_y = Input((self.target_dim,))
        inputs_u = Input((self.u_dim,))
        
        self.model_psi = Model(inputs=inputs_x, outputs=self.dic.call(inputs_x))
        psi_x = self.model_psi(inputs_x)
        psi_y = self.model_psi(inputs_y)
        
        self.model_k = Model(inputs=inputs_u,outputs=self.knn.call(inputs_u))
        psi_x = tf.expand_dims(psi_x, 1)
        psi_y = tf.expand_dims(psi_y, 1)
        outputs = tf.matmul(psi_x, self.model_k(inputs_u))-psi_y
        outputs = tf.reshape(outputs,(tf.shape(psi_x)[0],-1))
        self.model_KoopmanDL = Model(inputs=[inputs_x, inputs_y, inputs_u], outputs=outputs)
        return
            
    def Train_Dic(self):
        self.model_k.trainable = False
        self.model_psi.trainable = True
        return 
    
    def Train_Operator(self):
        self.model_k.trainable = True
        self.model_psi.trainable = False
        return
    
    def Train_OutputLayer(self):
        self.k_input = self.model_k.get_layer('K_input')
        self.k_input.trainable = False
        self.k_hidden_0 = self.model_k.get_layer('K_hidden_0')
        self.k_hidden_0.trainable = False
        self.k_hidden_1 = self.model_k.get_layer('K_hidden_1')
        self.k_hidden_1.trainable = False
        self.k_output = self.model_k.get_layer('K_output')
        self.k_output.trainable = True
        self.model_psi.trainable = False
        return
    
    def Predict(self, inputs_x, inputs_u):
        psi_x = self.model_psi(inputs_x)
        psi_x = tf.expand_dims(psi_x, 1)
        outputs = tf.matmul(psi_x, self.model_k(inputs_u))
        outputs = tf.reshape(outputs,(tf.shape(psi_x)[0],-1))
        return outputs[:,1:5]
    
    def Long_time_predict(self, x0, steps, inputs):
        predict_results = [x0]
        inputs_scaler = self.transform_u(inputs)
        for step in range(1, steps + 1):
            x0_scaler = self.transform_x(x0)
            x1_scaler = self.Predict(x0_scaler, inputs_scaler[step-1,:])
            x1 = self.inverse_transform_y(x1_scaler)
            predict_results.append(x1)
            x0 = x1
        return predict_results
    
    def get_loss(self, inputs_x, inputs_y, inputs_u, zeros_data_y, reg_para = 0.0001):
        weights = self.k_output.get_weights()[0]
        outputs = self.model_KoopmanDL([inputs_x, inputs_y, inputs_u])
        return self.loss_fun(outputs, zeros_data_y) + reg_para * tf.norm(weights)
    
    def train_model(self, train_data, test_data, epoches, reg_para):
        # Prepare dataset
        x_train, y_train, u_train = train_data
        x_test, y_test, u_test = test_data
        zeros_train = tf.zeros_like(self.dic.call(y_train))
        zeros_test = tf.zeros_like(self.dic.call(y_test))
        batch_size = 4096
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, u_train, zeros_train))
        # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test, u_test, zeros_test))
        train_dataset = train_dataset.shuffle(buffer_size = 8192).batch(batch_size)
        # test_dataset = test_dataset.batch(batch_size)
        
        for epoch in range(epoches):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            for step, (x_batch_train, y_batch_train, u_batch_train, zeros_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    loss_value = self.get_loss(x_batch_train, y_batch_train, u_batch_train, zeros_batch_train, reg_para)
                grads = tape.gradient(loss_value, self.model_KoopmanDL.trainable_weights)
                tf.keras.optimizers.Adam().apply_gradients(zip(grads, self.model_KoopmanDL.trainable_weights))
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.10f"
                        % (step, float(loss_value))
                        )
                    print("Seen so far: %d samples" % ((step + 1) * batch_size))
            
            test_loss_value = self.get_loss(x_test, y_test, u_test, zeros_test, reg_para)
            print("Testing loss at epoch %d: %.10f" % (epoch, float(test_loss_value)))
            print("Time taken: %.2fs" % (time.time() - start_time))
            