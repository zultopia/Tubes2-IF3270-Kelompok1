class BaseLayer:
    def __init__(self):
        self.weights = {}
        self.biases = {}
        self.cache = {}

    def forward(self, x, training=False):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError