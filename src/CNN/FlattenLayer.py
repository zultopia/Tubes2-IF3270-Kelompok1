from CNN.BaseLayer import BaseLayer

class FlattenLayer(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, training=False):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
    
    def backward(self, grad_output):
        if 'input_shape' not in self.cache:
            raise ValueError("Forward pass harus dilakukan terlebih dahulu")
        
        original_shape = self.cache['input_shape']
        return grad_output.reshape(original_shape)

# class FlattenLayerWithBackward(FlattenLayer):
#     def backward(self, grad_output):
#         if 'input_shape' not in self.cache:
#             raise ValueError("Forward pass harus dilakukan terlebih dahulu")
        
#         original_shape = self.cache['input_shape']
#         return grad_output.reshape(original_shape)
    
#     def forward(self, x, training=False):
#         self.cache['input_shape'] = x.shape
#         return super().forward(x, training)
