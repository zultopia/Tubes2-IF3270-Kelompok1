from RNN.BaseLayer import BaseLayer
import numpy as np

class DropoutLayer(BaseLayer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
    
    def forward(self, x, training=False):
        if not training:
            return x
        
        mask = np.random.binomial(1, 1 - self.rate, x.shape) / (1 - self.rate)
        self.cache['mask'] = mask
        return x * mask
    
    def backward(self, grad_output):
        if 'mask' in self.cache:
            return grad_output * self.cache['mask']
        return grad_output