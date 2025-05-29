from RNN.BaseLayer import BaseLayer
import numpy as np
from utils.Activation import Activation

class DenseLayer(BaseLayer):
    def __init__(self, units, activation='softmax', weights=None):
        super().__init__()
        self.units = units
        self.activation_name = activation
        self.activation = getattr(Activation, activation)
        self.d_activation = getattr(Activation, 'd_' + activation)

        if weights is not None:
            self.weights = {
                'kernel': weights[0],
                'bias': weights[1]
            }
        else:
            self.weights = {}
            self.biases = {}

        self.cache = {}

    def build(self, input_shape):
        if 'kernel' not in self.weights:
            fan_in = input_shape[1]
            limit = np.sqrt(6.0 / (fan_in + self.units))
            self.weights['kernel'] = np.random.uniform(-limit, limit, (fan_in, self.units))
            self.weights['bias'] = np.zeros((1, self.units))

    def forward(self, x, training=False):
        # (batch, seq, features)
        if len(x.shape) == 3: 
            x = x[:, -1, :] 

        self.build(x.shape)
        self.cache['input'] = x

        z = np.dot(x, self.weights['kernel']) + self.weights['bias']
        self.cache['linear_output'] = z

        a = self.activation(z)
        self.cache['output'] = a
        return a

    def backward(self, grad_output):
        x = self.cache['input']
        z = self.cache['linear_output']

        if self.activation_name == 'softmax':
            dz = grad_output
        else:
            dz = grad_output * self.d_activation(z)

        dw = np.dot(x.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        dx = np.dot(dz, self.weights['kernel'].T)

        self.gradients = {
            'kernel': dw,
            'bias': db
        }
        return dx
