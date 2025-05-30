import numpy as np
from CNN.BaseLayer import BaseLayer


class DenseLayer(BaseLayer):
    def __init__(self, input_size=None, output_size=None, activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.initialized = False
        self.weights = {'kernel': None}
        self.biases = {'bias': None}
        self.gradients = {'kernel': None, 'bias': None}
        
        if input_size is not None and output_size is not None:
            self._initialize_weights()
    
    def _initialize_weights(self):
        if self.input_size is None or self.output_size is None:
            raise ValueError("Input size dan output size harus ditentukan sebelum inisialisasi")
        
        std = np.sqrt(2.0 / self.input_size)
        self.weights['kernel'] = np.random.normal(0, std, (self.input_size, self.output_size))
        self.biases['bias'] = np.zeros((1, self.output_size))
        self.initialized = True
    
    def build(self, input_shape):
        if not self.initialized and len(input_shape) >= 2:
            self.input_size = input_shape[1]
            if self.output_size is not None:
                self._initialize_weights()
    
    def _apply_activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            return x
    
    def forward(self, x, training=False):
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        
        self.cache['input'] = x
        
        if not self.initialized:
            self.build(x.shape)
        
        linear_output = np.dot(x, self.weights['kernel']) + self.biases['bias']
        self.cache['linear_output'] = linear_output
        
        output = self._apply_activation(linear_output)
        self.cache['output'] = output
        
        return output
    
    def backward(self, grad_output):
        if 'input' not in self.cache:
            raise ValueError("Forward pass harus dilakukan terlebih dahulu")
        
        input_data = self.cache['input']
        linear_output = self.cache.get('linear_output', None)
        
        if self.activation == 'relu':
            grad_activation = (linear_output > 0).astype(float)
            grad_linear = grad_output * grad_activation
        elif self.activation == 'softmax':
            grad_linear = grad_output
        elif self.activation == 'sigmoid':
            sigmoid_output = self.cache['output']
            grad_linear = grad_output * sigmoid_output * (1 - sigmoid_output)
        elif self.activation == 'tanh':
            tanh_output = self.cache['output']
            grad_linear = grad_output * (1 - tanh_output**2)
        else:
            grad_linear = grad_output
        
        self.gradients['kernel'] = np.dot(input_data.T, grad_linear)
        self.gradients['bias'] = np.sum(grad_linear, axis=0, keepdims=True)
        
        grad_input = np.dot(grad_linear, self.weights['kernel'].T)
        
        return grad_input
    
    def set_weights(self, weights, biases):
        self.weights['kernel'] = weights
        self.biases['bias'] = biases.reshape(1, -1) if biases.ndim == 1 else biases
        self.initialized = True
        
        if self.input_size is None:
            self.input_size = weights.shape[0]
        if self.output_size is None:
            self.output_size = weights.shape[1]