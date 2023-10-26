from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.layers import Input, Add, Multiply, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from scipy.cluster.vq import kmeans
import scipy
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
        layer_sizes=[64, 64],
        n_psi_train=18,
        **kwargs):
        super(KoopmanDL_PsiNN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.dic_trainable = dic_trainable
        self.n_dic_customized = n_psi_train
        self.dicNN = self.dic_trainable(
            layer_sizes=self.layer_sizes,
            n_psi_train=self.n_dic_customized)
        self.n_psi_train = n_psi_train
    
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
    def __init__(self, n_K = 27, layer_sizes=[32, 32], **kwargs):
        self.layer_sizes = layer_sizes
        self.n_K = n_K
        self.input_layer = Dense(self.layer_sizes[0])
        self.hidden_layers = [Dense(layer_sizes[i], activation='tanh') for i in range(len(layer_sizes))]
        self.output_layer = Dense(self.n_K**2)
        
    def call(self, inputs_u):
        K = self.input_layer(inputs_u)
        for layer in self.hidden_layers:
            K = K + layer(K)
        outputs = self.output_layer(K)
        return tf.reshape(outputs,(-1,self.n_K,self.n_K))
 