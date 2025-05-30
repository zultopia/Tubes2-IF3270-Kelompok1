import numpy as np
from CNN.BaseLayer import BaseLayer


class MaxPooling2D(BaseLayer):    
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size
        self.padding = padding
    
    def _calculate_output_shape(self, input_shape):
        h, w, c = input_shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        if self.padding == 'same':
            oh = int(np.ceil(h / sh))
            ow = int(np.ceil(w / sw))
        else:
            oh = int((h - ph) / sh) + 1
            ow = int((w - pw) / sw) + 1
        
        return (oh, ow, c)
    
    def forward(self, x, training=False):
        batch_size, h, w, c = x.shape
        oh, ow, _ = self._calculate_output_shape((h, w, c))
        output = np.zeros((batch_size, oh, ow, c))
        
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        for b in range(batch_size):
            for ch in range(c):
                for i in range(oh):
                    for j in range(ow):
                        h_start = i * sh
                        h_end = min(h_start + ph, h)
                        w_start = j * sw
                        w_end = min(w_start + pw, w)
                        
                        if h_end > h_start and w_end > w_start:
                            output[b, i, j, ch] = np.max(x[b, h_start:h_end, w_start:w_end, ch])
        
        return output
    
    def backward(self, grad_output):
        if 'input' not in self.cache or 'mask' not in self.cache:
            raise ValueError("Forward pass harus dilakukan terlebih dahulu")
        
        input_data = self.cache['input']
        mask = self.cache['mask']
        
        grad_input = np.zeros_like(input_data)
        batch_size, oh, ow, c = grad_output.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        for b in range(batch_size):
            for ch in range(c):
                for i in range(oh):
                    for j in range(ow):
                        h_start = i * sh
                        h_end = min(h_start + ph, input_data.shape[1])
                        w_start = j * sw
                        w_end = min(w_start + pw, input_data.shape[2])
                        
                        max_mask = mask[b, i, j, ch]
                        if max_mask is not None:
                            max_h, max_w = max_mask
                            grad_input[b, max_h, max_w, ch] += grad_output[b, i, j, ch]
        
        return grad_input


class AveragePooling2D(BaseLayer):    
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size
        self.padding = padding
    
    def _calculate_output_shape(self, input_shape):
        h, w, c = input_shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        if self.padding == 'same':
            oh = int(np.ceil(h / sh))
            ow = int(np.ceil(w / sw))
        else:
            oh = int((h - ph) / sh) + 1
            ow = int((w - pw) / sw) + 1
        
        return (oh, ow, c)
    
    def forward(self, x, training=False):
        batch_size, h, w, c = x.shape
        oh, ow, _ = self._calculate_output_shape((h, w, c))
        output = np.zeros((batch_size, oh, ow, c))
        
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        for b in range(batch_size):
            for ch in range(c):
                for i in range(oh):
                    for j in range(ow):
                        h_start = i * sh
                        h_end = min(h_start + ph, h)
                        w_start = j * sw
                        w_end = min(w_start + pw, w)
                        
                        if h_end > h_start and w_end > w_start:
                            output[b, i, j, ch] = np.mean(x[b, h_start:h_end, w_start:w_end, ch])
        
        return output
    
    def backward(self, grad_output):
        if 'input' not in self.cache:
            raise ValueError("Forward pass harus dilakukan terlebih dahulu")
        
        input_data = self.cache['input']
        grad_input = np.zeros_like(input_data)
        
        batch_size, oh, ow, c = grad_output.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        for b in range(batch_size):
            for ch in range(c):
                for i in range(oh):
                    for j in range(ow):
                        h_start = i * sh
                        h_end = min(h_start + ph, input_data.shape[1])
                        w_start = j * sw
                        w_end = min(w_start + pw, input_data.shape[2])
                        
                        pool_size = (h_end - h_start) * (w_end - w_start)
                        if pool_size > 0:
                            grad_per_element = grad_output[b, i, j, ch] / pool_size
                            grad_input[b, h_start:h_end, w_start:w_end, ch] += grad_per_element
        
        return grad_input

# class MaxPooling2DWithBackward(MaxPooling2D):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def backward(self, grad_output):
#         if 'input' not in self.cache or 'mask' not in self.cache:
#             raise ValueError("Forward pass harus dilakukan terlebih dahulu")
        
#         input_data = self.cache['input']
#         mask = self.cache['mask']
        
#         grad_input = np.zeros_like(input_data)
#         batch_size, oh, ow, c = grad_output.shape
#         ph, pw = self.pool_size
#         sh, sw = self.strides
        
#         for b in range(batch_size):
#             for ch in range(c):
#                 for i in range(oh):
#                     for j in range(ow):
#                         h_start = i * sh
#                         h_end = min(h_start + ph, input_data.shape[1])
#                         w_start = j * sw
#                         w_end = min(w_start + pw, input_data.shape[2])
                        
#                         max_mask = mask[b, i, j, ch]
#                         if max_mask is not None:
#                             max_h, max_w = max_mask
#                             grad_input[b, max_h, max_w, ch] += grad_output[b, i, j, ch]
        
#         return grad_input
    
#     def forward(self, x, training=False):
#         self.cache['input'] = x
        
#         batch_size, h, w, c = x.shape
#         oh, ow, _ = self._calculate_output_shape((h, w, c))
#         output = np.zeros((batch_size, oh, ow, c))
        
#         mask = np.empty((batch_size, oh, ow, c), dtype=object)
        
#         ph, pw = self.pool_size
#         sh, sw = self.strides
        
#         for b in range(batch_size):
#             for ch in range(c):
#                 for i in range(oh):
#                     for j in range(ow):
#                         h_start = i * sh
#                         h_end = min(h_start + ph, h)
#                         w_start = j * sw
#                         w_end = min(w_start + pw, w)
                        
#                         if h_end > h_start and w_end > w_start:
#                             region = x[b, h_start:h_end, w_start:w_end, ch]
#                             max_val = np.max(region)
#                             output[b, i, j, ch] = max_val
                            
#                             max_pos = np.unravel_index(np.argmax(region), region.shape)
#                             mask[b, i, j, ch] = (h_start + max_pos[0], w_start + max_pos[1])
        
#         self.cache['mask'] = mask
#         return output

# class AveragePooling2DWithBackward(AveragePooling2D):
#     def backward(self, grad_output):
#         if 'input' not in self.cache:
#             raise ValueError("Forward pass harus dilakukan terlebih dahulu")
        
#         input_data = self.cache['input']
#         grad_input = np.zeros_like(input_data)
        
#         batch_size, oh, ow, c = grad_output.shape
#         ph, pw = self.pool_size
#         sh, sw = self.strides
        
#         for b in range(batch_size):
#             for ch in range(c):
#                 for i in range(oh):
#                     for j in range(ow):
#                         h_start = i * sh
#                         h_end = min(h_start + ph, input_data.shape[1])
#                         w_start = j * sw
#                         w_end = min(w_start + pw, input_data.shape[2])
                        
#                         pool_size = (h_end - h_start) * (w_end - w_start)
#                         if pool_size > 0:
#                             grad_per_element = grad_output[b, i, j, ch] / pool_size
#                             grad_input[b, h_start:h_end, w_start:w_end, ch] += grad_per_element
        
#         return grad_input
    
#     def forward(self, x, training=False):
#         self.cache['input'] = x
#         return super().forward(x, training)
