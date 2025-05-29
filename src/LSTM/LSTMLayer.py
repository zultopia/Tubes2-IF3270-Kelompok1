from RNN.BaseLayer import BaseLayer
import numpy as np

class LSTMLayer(BaseLayer):
    def __init__(self, units, return_sequences=True, weights=None):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.initialized = False
        
        if weights is not None:
            # LSTM weights format: [kernel, recurrent_kernel, bias]
            # kernel shape: (input_dim, 4 * units) - for i, f, c, o gates
            # recurrent_kernel shape: (units, 4 * units)
            # bias shape: (4 * units,)
            self.weights['kernel'] = weights[0]
            self.weights['recurrent_kernel'] = weights[1]  
            self.biases['bias'] = weights[2]
            self.initialized = True
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, x, training=False):
        batch_size, seq_len, input_dim = x.shape
        
        if not self.initialized:
            # Xavier initialization
            limit = np.sqrt(6.0 / (input_dim + self.units))
            self.weights['kernel'] = np.random.uniform(-limit, limit, (input_dim, 4 * self.units))
            
            limit_rec = np.sqrt(6.0 / (self.units + self.units))
            self.weights['recurrent_kernel'] = np.random.uniform(-limit_rec, limit_rec, (self.units, 4 * self.units))
            
            self.biases['bias'] = np.zeros(4 * self.units)
            self.initialized = True
        
        # Initialize hidden and cell states
        h = np.zeros((batch_size, self.units))
        c = np.zeros((batch_size, self.units))
        
        outputs = []
        hidden_states = [h.copy()]
        cell_states = [c.copy()]
        
        # Store intermediate values for backward pass
        gates_history = []
        
        for t in range(seq_len):
            # Compute gates
            gates = np.dot(x[:, t, :], self.weights['kernel']) + \
                   np.dot(h, self.weights['recurrent_kernel']) + \
                   self.biases['bias']
            
            # Split gates: input, forget, cell, output
            i_gate = self._sigmoid(gates[:, :self.units])                    # input gate
            f_gate = self._sigmoid(gates[:, self.units:2*self.units])        # forget gate  
            c_tilde = self._tanh(gates[:, 2*self.units:3*self.units])        # candidate values
            o_gate = self._sigmoid(gates[:, 3*self.units:])                  # output gate
            
            # Update cell state
            c = f_gate * c + i_gate * c_tilde
            
            # Update hidden state
            h = o_gate * self._tanh(c)
            
            outputs.append(h.copy())
            hidden_states.append(h.copy())
            cell_states.append(c.copy())
            
            # Store gates for backward pass
            gates_history.append({
                'i_gate': i_gate,
                'f_gate': f_gate,
                'c_tilde': c_tilde,
                'o_gate': o_gate,
                'gates': gates
            })
        
        # Cache for backward pass
        self.cache['input'] = x
        self.cache['hidden_states'] = hidden_states
        self.cache['cell_states'] = cell_states
        self.cache['outputs'] = outputs
        self.cache['gates_history'] = gates_history
        self.cache['batch_size'] = batch_size
        self.cache['seq_len'] = seq_len
        
        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return h
    
    def backward(self, grad_output):
        x = self.cache['input']
        hidden_states = self.cache['hidden_states']
        cell_states = self.cache['cell_states']
        outputs = self.cache['outputs']
        gates_history = self.cache['gates_history']
        batch_size = self.cache['batch_size']
        seq_len = self.cache['seq_len']
        
        # Initialize gradients
        grad_kernel = np.zeros_like(self.weights['kernel'])
        grad_recurrent_kernel = np.zeros_like(self.weights['recurrent_kernel'])
        grad_bias = np.zeros_like(self.biases['bias'])
        grad_input = np.zeros_like(x)
        
        # Initialize gradients for hidden and cell states
        if not self.return_sequences:
            grad_h_next = grad_output
            grad_c_next = np.zeros((batch_size, self.units))
        else:
            grad_h_next = np.zeros((batch_size, self.units))
            grad_c_next = np.zeros((batch_size, self.units))
        
        # Backward through time
        for t in range(seq_len - 1, -1, -1):
            if self.return_sequences:
                grad_h = grad_output[:, t, :] + grad_h_next
            else:
                if t == seq_len - 1:
                    grad_h = grad_output + grad_h_next
                else:
                    grad_h = grad_h_next
            
            # Get current states and gates
            c_t = cell_states[t + 1]
            c_prev = cell_states[t]
            h_prev = hidden_states[t]
            
            gates = gates_history[t]
            i_gate = gates['i_gate']
            f_gate = gates['f_gate']
            c_tilde = gates['c_tilde']
            o_gate = gates['o_gate']
            
            # Gradient w.r.t output gate
            grad_o = grad_h * self._tanh(c_t)
            grad_o_input = grad_o * o_gate * (1 - o_gate)
            
            # Gradient w.r.t cell state
            grad_c = grad_c_next + grad_h * o_gate * (1 - self._tanh(c_t)**2)
            
            # Gradient w.r.t candidate cell state
            grad_c_tilde = grad_c * i_gate
            grad_c_tilde_input = grad_c_tilde * (1 - c_tilde**2)
            
            # Gradient w.r.t input gate
            grad_i = grad_c * c_tilde
            grad_i_input = grad_i * i_gate * (1 - i_gate)
            
            # Gradient w.r.t forget gate
            grad_f = grad_c * c_prev
            grad_f_input = grad_f * f_gate * (1 - f_gate)
            
            # Combine gate gradients
            grad_gates = np.concatenate([
                grad_i_input, grad_f_input, grad_c_tilde_input, grad_o_input
            ], axis=1)
            
            # Gradients w.r.t weights and biases
            grad_kernel += np.dot(x[:, t, :].T, grad_gates)
            grad_recurrent_kernel += np.dot(h_prev.T, grad_gates)
            grad_bias += np.sum(grad_gates, axis=0)
            
            # Gradients w.r.t input
            grad_input[:, t, :] = np.dot(grad_gates, self.weights['kernel'].T)
            
            # Gradients for next timestep
            grad_h_next = np.dot(grad_gates, self.weights['recurrent_kernel'].T)
            grad_c_next = grad_c * f_gate
        
        self.gradients = {
            'kernel': grad_kernel,
            'recurrent_kernel': grad_recurrent_kernel,
            'bias': grad_bias
        }
        
        return grad_input