from RNN.BaseLayer import BaseLayer
import numpy as np

class BidirectionalLSTMLayer(BaseLayer):
    def __init__(self, units, return_sequences=True, weights=None):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.forward_units = units // 2
        self.backward_units = units // 2
        self.initialized = False
        
        if weights is not None:
            # Forward LSTM weights
            self.weights['forward_kernel'] = weights[0]
            self.weights['forward_recurrent'] = weights[1]
            self.biases['forward_bias'] = weights[2]

            # Backward LSTM weights
            self.weights['backward_kernel'] = weights[3]
            self.weights['backward_recurrent'] = weights[4]
            self.biases['backward_bias'] = weights[5]

            self.initialized = True
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, x, training=False):
        batch_size, seq_len, input_dim = x.shape

        if not self.initialized:
            for direction in ['forward', 'backward']:
                limit = np.sqrt(6.0 / (input_dim + self.forward_units))
                self.weights[f'{direction}_kernel'] = np.random.uniform(
                    -limit, limit, (input_dim, 4 * self.forward_units)
                )
                
                limit_rec = np.sqrt(6.0 / (self.forward_units + self.forward_units))
                self.weights[f'{direction}_recurrent'] = np.random.uniform(
                    -limit_rec, limit_rec, (self.forward_units, 4 * self.forward_units)
                )
                
                self.biases[f'{direction}_bias'] = np.zeros(4 * self.forward_units)
            self.initialized = True
        
        # Forward LSTM pass
        h_forward = np.zeros((batch_size, self.forward_units))
        c_forward = np.zeros((batch_size, self.forward_units))
        forward_outputs = []
        forward_hidden_states = [h_forward.copy()]
        forward_cell_states = [c_forward.copy()]
        forward_gates_history = []
        
        for t in range(seq_len):
            gates = np.dot(x[:, t, :], self.weights['forward_kernel']) + \
                   np.dot(h_forward, self.weights['forward_recurrent']) + \
                   self.biases['forward_bias']
            
            i_gate = self._sigmoid(gates[:, :self.forward_units])
            f_gate = self._sigmoid(gates[:, self.forward_units:2*self.forward_units])
            c_tilde = self._tanh(gates[:, 2*self.forward_units:3*self.forward_units])
            o_gate = self._sigmoid(gates[:, 3*self.forward_units:])
            
            c_forward = f_gate * c_forward + i_gate * c_tilde
            h_forward = o_gate * self._tanh(c_forward)
            
            forward_outputs.append(h_forward.copy())
            forward_hidden_states.append(h_forward.copy())
            forward_cell_states.append(c_forward.copy())
            
            forward_gates_history.append({
                'i_gate': i_gate, 'f_gate': f_gate, 
                'c_tilde': c_tilde, 'o_gate': o_gate, 'gates': gates
            })
        
        # Backward LSTM pass
        h_backward = np.zeros((batch_size, self.backward_units))
        c_backward = np.zeros((batch_size, self.backward_units))
        backward_outputs = []
        backward_hidden_states = [h_backward.copy()]
        backward_cell_states = [c_backward.copy()]
        backward_gates_history = []
        
        for t in range(seq_len - 1, -1, -1):
            gates = np.dot(x[:, t, :], self.weights['backward_kernel']) + \
                   np.dot(h_backward, self.weights['backward_recurrent']) + \
                   self.biases['backward_bias']
            
            i_gate = self._sigmoid(gates[:, :self.backward_units])
            f_gate = self._sigmoid(gates[:, self.backward_units:2*self.backward_units])
            c_tilde = self._tanh(gates[:, 2*self.backward_units:3*self.backward_units])
            o_gate = self._sigmoid(gates[:, 3*self.backward_units:])
            
            c_backward = f_gate * c_backward + i_gate * c_tilde
            h_backward = o_gate * self._tanh(c_backward)
            
            backward_outputs.insert(0, h_backward.copy())
            backward_hidden_states.insert(0, h_backward.copy())
            backward_cell_states.insert(0, c_backward.copy())
            
            backward_gates_history.insert(0, {
                'i_gate': i_gate, 'f_gate': f_gate,
                'c_tilde': c_tilde, 'o_gate': o_gate, 'gates': gates
            })
        
        # Cache for backward pass
        self.cache['input'] = x
        self.cache['forward_hidden_states'] = forward_hidden_states
        self.cache['forward_cell_states'] = forward_cell_states
        self.cache['forward_outputs'] = forward_outputs
        self.cache['forward_gates_history'] = forward_gates_history
        self.cache['backward_hidden_states'] = backward_hidden_states
        self.cache['backward_cell_states'] = backward_cell_states
        self.cache['backward_outputs'] = backward_outputs
        self.cache['backward_gates_history'] = backward_gates_history
        self.cache['batch_size'] = batch_size
        self.cache['seq_len'] = seq_len
        
        # Concatenate forward and backward outputs
        if self.return_sequences:
            forward_stack = np.stack(forward_outputs, axis=1)
            backward_stack = np.stack(backward_outputs, axis=1)
            return np.concatenate([forward_stack, backward_stack], axis=-1)
        else:
            return np.concatenate([forward_outputs[-1], backward_outputs[0]], axis=-1)
    
    def backward(self, grad_output):
        x = self.cache['input']
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
        
        # Backward pass for forward LSTM
        grad_h_next = np.zeros((batch_size, self.forward_units))
        grad_c_next = np.zeros((batch_size, self.forward_units))
        
        for t in range(seq_len - 1, -1, -1):
            if self.return_sequences:
                grad_h = grad_forward[:, t, :] + grad_h_next
            else:
                if t == seq_len - 1:
                    grad_h = grad_forward + grad_h_next
                else:
                    grad_h = grad_h_next
            
            # Get states and gates for forward LSTM
            c_t = self.cache['forward_cell_states'][t + 1]
            c_prev = self.cache['forward_cell_states'][t]
            h_prev = self.cache['forward_hidden_states'][t]
            gates = self.cache['forward_gates_history'][t]
            
            # Compute gradients (same logic as LSTMLayer)
            grad_o = grad_h * self._tanh(c_t)
            grad_o_input = grad_o * gates['o_gate'] * (1 - gates['o_gate'])
            
            grad_c = grad_c_next + grad_h * gates['o_gate'] * (1 - self._tanh(c_t)**2)
            
            grad_c_tilde = grad_c * gates['i_gate']
            grad_c_tilde_input = grad_c_tilde * (1 - gates['c_tilde']**2)
            
            grad_i = grad_c * gates['c_tilde']
            grad_i_input = grad_i * gates['i_gate'] * (1 - gates['i_gate'])
            
            grad_f = grad_c * c_prev
            grad_f_input = grad_f * gates['f_gate'] * (1 - gates['f_gate'])
            
            grad_gates = np.concatenate([
                grad_i_input, grad_f_input, grad_c_tilde_input, grad_o_input
            ], axis=1)
            
            grad_forward_kernel += np.dot(x[:, t, :].T, grad_gates)
            grad_forward_recurrent += np.dot(h_prev.T, grad_gates)
            grad_forward_bias += np.sum(grad_gates, axis=0)
            
            grad_input[:, t, :] += np.dot(grad_gates, self.weights['forward_kernel'].T)
            grad_h_next = np.dot(grad_gates, self.weights['forward_recurrent'].T)
            grad_c_next = grad_c * gates['f_gate']
        
        # Backward pass for backward LSTM
        grad_h_next = np.zeros((batch_size, self.backward_units))
        grad_c_next = np.zeros((batch_size, self.backward_units))
        
        for t in range(seq_len):
            if self.return_sequences:
                grad_h = grad_backward[:, t, :] + grad_h_next
            else:
                if t == 0:
                    grad_h = grad_backward + grad_h_next
                else:
                    grad_h = grad_h_next
            
            # Get states and gates for backward LSTM
            c_t = self.cache['backward_cell_states'][t + 1]
            c_prev = self.cache['backward_cell_states'][t]
            h_prev = self.cache['backward_hidden_states'][t]
            gates = self.cache['backward_gates_history'][t]
            
            # Compute gradients (same logic as LSTMLayer)
            grad_o = grad_h * self._tanh(c_t)
            grad_o_input = grad_o * gates['o_gate'] * (1 - gates['o_gate'])
            
            grad_c = grad_c_next + grad_h * gates['o_gate'] * (1 - self._tanh(c_t)**2)
            
            grad_c_tilde = grad_c * gates['i_gate']
            grad_c_tilde_input = grad_c_tilde * (1 - gates['c_tilde']**2)
            
            grad_i = grad_c * gates['c_tilde']
            grad_i_input = grad_i * gates['i_gate'] * (1 - gates['i_gate'])
            
            grad_f = grad_c * c_prev
            grad_f_input = grad_f * gates['f_gate'] * (1 - gates['f_gate'])
            
            grad_gates = np.concatenate([
                grad_i_input, grad_f_input, grad_c_tilde_input, grad_o_input
            ], axis=1)
            
            grad_backward_kernel += np.dot(x[:, t, :].T, grad_gates)
            grad_backward_recurrent += np.dot(h_prev.T, grad_gates)
            grad_backward_bias += np.sum(grad_gates, axis=0)
            
            grad_input[:, t, :] += np.dot(grad_gates, self.weights['backward_kernel'].T)
            grad_h_next = np.dot(grad_gates, self.weights['backward_recurrent'].T)
            grad_c_next = grad_c * gates['f_gate']
        
        self.gradients = {
            'forward_kernel': grad_forward_kernel,
            'forward_recurrent': grad_forward_recurrent,
            'forward_bias': grad_forward_bias,
            'backward_kernel': grad_backward_kernel,
            'backward_recurrent': grad_backward_recurrent,
            'backward_bias': grad_backward_bias
        }
        
        return grad_input