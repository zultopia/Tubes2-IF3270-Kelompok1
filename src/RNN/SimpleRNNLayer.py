from RNN.BaseLayer import BaseLayer
import numpy as np

class SimpleRNNLayer(BaseLayer):
    def __init__(self, units, return_sequences=True, weights=None):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.initialized = False
        
        if weights is not None:
            self.weights['kernel'] = weights[0]
            self.weights['recurrent_kernel'] = weights[1]
            self.biases['bias'] = weights[2]

            self.initialized = True
    
    def forward(self, x, training=False):
        batch_size, seq_len, input_dim = x.shape

        if not self.initialized:
            self.weights['kernel'] = np.random.uniform(
                -np.sqrt(6.0 / (self.units + self.units)), np.sqrt(6.0 / (self.units + self.units)),
                (input_dim, self.units)
            )
            self.weights['recurrent_kernel'] = np.random.uniform(
                -np.sqrt(6.0 / (self.units + self.units)), np.sqrt(6.0 / (self.units + self.units)),
                (self.units, self.units)
            )
            self.biases['bias'] = np.zeros(self.units)
            
            self.initialized = True
        
        h = np.zeros((batch_size, self.units))
        outputs = []
        hidden_states = [h.copy()]
        
        for t in range(seq_len):
            # h_t = tanh(x_t @ W + h_{t-1} @ U + b)
            linear = np.dot(x[:, t, :], self.weights['kernel']) + \
                    np.dot(h, self.weights['recurrent_kernel']) + self.biases['bias']
            h = np.tanh(linear)
            outputs.append(h.copy())
            hidden_states.append(h.copy())
        
        # Cache for backward pass
        self.cache['input'] = x
        self.cache['hidden_states'] = hidden_states
        self.cache['outputs'] = outputs
        self.cache['batch_size'] = batch_size
        self.cache['seq_len'] = seq_len
        
        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return h
    
    def backward(self, grad_output):
        x = self.cache['input']
        hidden_states = self.cache['hidden_states']
        outputs = self.cache['outputs']
        batch_size = self.cache['batch_size']
        seq_len = self.cache['seq_len']
        
        # Initialize gradients
        grad_kernel = np.zeros_like(self.weights['kernel'])
        grad_recurrent_kernel = np.zeros_like(self.weights['recurrent_kernel'])
        grad_bias = np.zeros_like(self.biases['bias'])
        grad_input = np.zeros_like(x)
        
        if not self.return_sequences:
            grad_h = grad_output
            grad_h_next = np.zeros_like(grad_h)
        else:
            grad_h_next = np.zeros((batch_size, self.units))
        
        # Backward through time
        for t in range(seq_len - 1, -1, -1):
            if self.return_sequences:
                grad_h = grad_output[:, t, :] + grad_h_next
            else:
                if t == seq_len - 1:
                    grad_h = grad_output + grad_h_next
                else:
                    grad_h = grad_h_next
            
            # Gradient
            h_t = outputs[t]
            grad_tanh = grad_h * (1 - h_t ** 2)
            grad_kernel += np.dot(x[:, t, :].T, grad_tanh)
            grad_recurrent_kernel += np.dot(hidden_states[t].T, grad_tanh)
            grad_bias += np.sum(grad_tanh, axis=0)
            grad_input[:, t, :] = np.dot(grad_tanh, self.weights['kernel'].T)
            grad_h_next = np.dot(grad_tanh, self.weights['recurrent_kernel'].T)
        
        self.gradients = {
            'kernel': grad_kernel,
            'recurrent_kernel': grad_recurrent_kernel,
            'bias': grad_bias
        }
        
        return grad_input
