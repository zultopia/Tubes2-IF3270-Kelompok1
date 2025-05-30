import numpy as np
from CNN.BaseLayer import BaseLayer


class Conv2DLayer(BaseLayer):
    def __init__(self, input_shape, filters, kernel_size=(3, 3), strides=(1, 1), 
                 padding='same', activation='relu'):
        super().__init__()
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.initialized = False
        self.gradients = {'kernel': None, 'bias': None}
        
        self._initialize_weights()
        self.output_shape = self._calculate_output_shape()
    
    def _initialize_weights(self):
        h, w, c = self.input_shape
        kh, kw = self.kernel_size
        fan_in = kh * kw * c
        std = np.sqrt(2.0 / fan_in) 
        
        self.weights['kernel'] = np.random.normal(0, std, (kh, kw, c, self.filters))
        self.biases['bias'] = np.zeros(self.filters)
        self.initialized = True
    
    def _calculate_output_shape(self):
        h, w, c = self.input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides
        
        if self.padding == 'same':
            oh = int(np.ceil(h / sh))
            ow = int(np.ceil(w / sw))
        else:
            oh = int((h - kh) / sh) + 1
            ow = int((w - kw) / sw) + 1
        
        return (oh, ow, self.filters)
    
    def _add_padding(self, x):
        if self.padding == 'same':
            h, w = x.shape[1], x.shape[2]
            kh, kw = self.kernel_size
            sh, sw = self.strides
            
            pad_h = max(0, (self.output_shape[0] - 1) * sh + kh - h)
            pad_w = max(0, (self.output_shape[1] - 1) * sw + kw - w)
            
            pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
            pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
            
            x = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
        return x
    
    def _apply_activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x
    
    def forward(self, x, training=False):
        self.cache['input'] = x
        
        batch_size = x.shape[0]
        x_padded = self._add_padding(x)
        
        oh, ow, _ = self.output_shape
        output = np.zeros((batch_size, oh, ow, self.filters))
        
        kh, kw = self.kernel_size
        sh, sw = self.strides
        
        for b in range(batch_size):
            for f in range(self.filters):
                for h in range(oh):
                    for w in range(ow):
                        h_start = h * sh
                        h_end = h_start + kh
                        w_start = w * sw
                        w_end = w_start + kw
                        
                        if h_end <= x_padded.shape[1] and w_end <= x_padded.shape[2]:
                            region = x_padded[b, h_start:h_end, w_start:w_end, :]
                            conv_result = np.sum(region * self.weights['kernel'][:, :, :, f])
                            output[b, h, w, f] = conv_result + self.biases['bias'][f]
        
        return self._apply_activation(output)
    
    def backward(self, grad_output):
        if 'input' not in self.cache:
            raise ValueError("Forward pass must be performed first")
        
        input_data = self.cache['input']
        batch_size, input_h, input_w, input_c = input_data.shape
        _, output_h, output_w, output_c = grad_output.shape
        
        grad_kernel = np.zeros_like(self.weights['kernel'])
        grad_bias = np.zeros_like(self.biases['bias'])
        grad_input = np.zeros_like(input_data)
        
        input_padded = self._add_padding(input_data)
        kh, kw = self.kernel_size
        sh, sw = self.strides
        
        for b in range(batch_size):
            for f in range(self.filters):
                for h in range(output_h):
                    for w in range(output_w):
                        h_start = h * sh
                        h_end = h_start + kh
                        w_start = w * sw
                        w_end = w_start + kw
                        
                        if h_end <= input_padded.shape[1] and w_end <= input_padded.shape[2]:
                            region = input_padded[b, h_start:h_end, w_start:w_end, :]
                            
                            grad_kernel[:, :, :, f] += region * grad_output[b, h, w, f]
                            
                            grad_bias[f] += grad_output[b, h, w, f]
                            
                            grad_input_region = self.weights['kernel'][:, :, :, f] * grad_output[b, h, w, f]
                            
                            if self.padding == 'same':
                                pad_h = max(0, (output_h - 1) * sh + kh - input_h)
                                pad_w = max(0, (output_w - 1) * sw + kw - input_w)
                                pad_top = pad_h // 2
                                pad_left = pad_w // 2
                                
                                input_h_start = max(0, h_start - pad_top)
                                input_h_end = min(input_h, h_end - pad_top)
                                input_w_start = max(0, w_start - pad_left)
                                input_w_end = min(input_w, w_end - pad_left)
                                
                                if (input_h_start < input_h_end and input_w_start < input_w_end):
                                    kernel_h_start = max(0, pad_top - h_start)
                                    kernel_h_end = kernel_h_start + (input_h_end - input_h_start)
                                    kernel_w_start = max(0, pad_left - w_start)
                                    kernel_w_end = kernel_w_start + (input_w_end - input_w_start)
                                    
                                    grad_input[b, input_h_start:input_h_end, input_w_start:input_w_end, :] += \
                                        grad_input_region[kernel_h_start:kernel_h_end, kernel_w_start:kernel_w_end, :]
                            else:
                                grad_input[b, h_start:h_end, w_start:w_end, :] += grad_input_region
        
        self.gradients['kernel'] = grad_kernel
        self.gradients['bias'] = grad_bias
        
        return grad_input
    
    def set_weights(self, weights, biases):
        self.weights['kernel'] = weights
        self.biases['bias'] = biases