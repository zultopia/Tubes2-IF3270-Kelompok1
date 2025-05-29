from RNN.BaseLayer import BaseLayer
import numpy as np

class BidirectionalRNNLayer(BaseLayer):
    def __init__(self, units, return_sequences=True, weights=None):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.forward_units = units // 2
        self.backward_units = units // 2
        self.initialized = False
        
        if weights is not None:
            # Forward RNN weights
            self.weights['forward_kernel'] = weights[0]
            self.weights['forward_recurrent'] = weights[1]
            self.biases['forward_bias'] = weights[2]

            # Backward RNN weights
            self.weights['backward_kernel'] = weights[3]
            self.weights['backward_recurrent'] = weights[4]
            self.biases['backward_bias'] = weights[5]

            self.initialized = True
        # else:
        #     # Initialize weights
        #     for direction in ['forward', 'backward']:
        #         self.weights[f'{direction}_kernel'] = np.random.uniform(
        #             -np.sqrt(6.0 / (self.forward_units + self.forward_units)),
        #             np.sqrt(6.0 / (self.forward_units + self.forward_units)),
        #             (self.forward_units, self.forward_units)
        #         )
        #         self.weights[f'{direction}_recurrent'] = np.random.uniform(
        #             -np.sqrt(6.0 / (self.forward_units + self.forward_units)),
        #             np.sqrt(6.0 / (self.forward_units + self.forward_units)),
        #             (self.forward_units, self.forward_units)
        #         )
        #         self.biases[f'{direction}_bias'] = np.zeros(self.forward_units)
    
    def forward(self, x, training=False):
        batch_size, seq_len, input_dim = x.shape

        if not self.initialized:
            for direction in ['forward', 'backward']:
                self.weights[f'{direction}_kernel'] = np.random.uniform(
                    -np.sqrt(6.0 / (self.forward_units + self.forward_units)),
                    np.sqrt(6.0 / (self.forward_units + self.forward_units)),
                    (input_dim, self.forward_units)
                )
                self.weights[f'{direction}_recurrent'] = np.random.uniform(
                    -np.sqrt(6.0 / (self.forward_units + self.forward_units)),
                    np.sqrt(6.0 / (self.forward_units + self.forward_units)),
                    (self.forward_units, self.forward_units)
                )
                self.biases[f'{direction}_bias'] = np.zeros(self.forward_units)

            # limit = np.sqrt(6.0 / (input_dim + self.forward_units))
            
            # self.weights['forward_kernel'] = np.random.uniform(-limit, limit, (input_dim, self.forward_units))
            # self.weights['forward_recurrent'] = np.random.uniform(-limit, limit, (self.forward_units, self.forward_units))
            # self.biases['forward_bias'] = np.zeros(self.forward_units)

            # self.weights['backward_kernel'] = np.random.uniform(-limit, limit, (input_dim, self.backward_units))
            # self.weights['backward_recurrent'] = np.random.uniform(-limit, limit, (self.backward_units, self.backward_units))
            # self.biases['backward_bias'] = np.zeros(self.backward_units)

            self.initialized = True
        
        # Forward pass
        h_forward = np.zeros((batch_size, self.forward_units))
        forward_outputs = []
        forward_hidden_states = [h_forward.copy()]
        
        for t in range(seq_len):
            linear = np.dot(x[:, t, :], self.weights['forward_kernel']) + \
                    np.dot(h_forward, self.weights['forward_recurrent']) + \
                    self.biases['forward_bias']
            h_forward = np.tanh(linear)
            forward_outputs.append(h_forward.copy())
            forward_hidden_states.append(h_forward.copy())
        
        # Backward pass
        h_backward = np.zeros((batch_size, self.backward_units))
        backward_outputs = []
        backward_hidden_states = [h_backward.copy()]
        
        for t in range(seq_len - 1, -1, -1):
            linear = np.dot(x[:, t, :], self.weights['backward_kernel']) + \
                    np.dot(h_backward, self.weights['backward_recurrent']) + \
                    self.biases['backward_bias']
            h_backward = np.tanh(linear)
            backward_outputs.insert(0, h_backward.copy())
            backward_hidden_states.insert(0, h_backward.copy())
        
        # Cache for backward pass
        self.cache['input'] = x
        self.cache['forward_hidden_states'] = forward_hidden_states
        self.cache['backward_hidden_states'] = backward_hidden_states
        self.cache['forward_outputs'] = forward_outputs
        self.cache['backward_outputs'] = backward_outputs
        self.cache['batch_size'] = batch_size
        self.cache['seq_len'] = seq_len
        
        # Concat forward and backward outputs
        if self.return_sequences:
            forward_stack = np.stack(forward_outputs, axis=1)
            backward_stack = np.stack(backward_outputs, axis=1)
            return np.concatenate([forward_stack, backward_stack], axis=-1)
        else:
            return np.concatenate([forward_outputs[-1], backward_outputs[0]], axis=-1)
    
    def backward(self, grad_output):
        x = self.cache['input']
        forward_hidden_states = self.cache['forward_hidden_states']
        backward_hidden_states = self.cache['backward_hidden_states']
        forward_outputs = self.cache['forward_outputs']
        backward_outputs = self.cache['backward_outputs']
        batch_size = self.cache['batch_size']
        seq_len = self.cache['seq_len']
        
        # Split gradient into forward and backward parts
        if self.return_sequences:
            grad_forward = grad_output[:, :, :self.forward_units]
            grad_backward = grad_output[:, :, self.forward_units:]
        else:
            grad_forward = grad_output[:, :self.forward_units]
            grad_backward = grad_output[:, self.forward_units:]
        
        # Initialize gradients
        grad_input = np.zeros_like(x)
        
        # Forward direction gradients
        grad_forward_kernel = np.zeros_like(self.weights['forward_kernel'])
        grad_forward_recurrent = np.zeros_like(self.weights['forward_recurrent'])
        grad_forward_bias = np.zeros_like(self.biases['forward_bias'])
        
        # Backward direction gradients
        grad_backward_kernel = np.zeros_like(self.weights['backward_kernel'])
        grad_backward_recurrent = np.zeros_like(self.weights['backward_recurrent'])
        grad_backward_bias = np.zeros_like(self.biases['backward_bias'])
        
        # Backward pass for forward RNN
        grad_h_next = np.zeros((batch_size, self.forward_units))
        for t in range(seq_len - 1, -1, -1):
            if self.return_sequences:
                grad_h = grad_forward[:, t, :] + grad_h_next
            else:
                if t == seq_len - 1:
                    grad_h = grad_forward + grad_h_next
                else:
                    grad_h = grad_h_next
            
            h_t = forward_outputs[t]
            grad_tanh = grad_h * (1 - h_t ** 2)
            
            grad_forward_kernel += np.dot(x[:, t, :].T, grad_tanh)
            grad_forward_recurrent += np.dot(forward_hidden_states[t].T, grad_tanh)
            grad_forward_bias += np.sum(grad_tanh, axis=0)
            
            grad_input[:, t, :] += np.dot(grad_tanh, self.weights['forward_kernel'].T)
            grad_h_next = np.dot(grad_tanh, self.weights['forward_recurrent'].T)
        
        # Backward pass for backward RNN
        grad_h_next = np.zeros((batch_size, self.backward_units))
        for t in range(seq_len):
            if self.return_sequences:
                grad_h = grad_backward[:, t, :] + grad_h_next
            else:
                if t == 0:
                    grad_h = grad_backward + grad_h_next
                else:
                    grad_h = grad_h_next
            
            h_t = backward_outputs[t]
            grad_tanh = grad_h * (1 - h_t ** 2)
            
            grad_backward_kernel += np.dot(x[:, t, :].T, grad_tanh)
            grad_backward_recurrent += np.dot(backward_hidden_states[t].T, grad_tanh)
            grad_backward_bias += np.sum(grad_tanh, axis=0)
            
            grad_input[:, t, :] += np.dot(grad_tanh, self.weights['backward_kernel'].T)
            grad_h_next = np.dot(grad_tanh, self.weights['backward_recurrent'].T)
        
        self.gradients = {
            'forward_kernel': grad_forward_kernel,
            'forward_recurrent': grad_forward_recurrent,
            'forward_bias': grad_forward_bias,
            'backward_kernel': grad_backward_kernel,
            'backward_recurrent': grad_backward_recurrent,
            'backward_bias': grad_backward_bias
        }
        
        return grad_input