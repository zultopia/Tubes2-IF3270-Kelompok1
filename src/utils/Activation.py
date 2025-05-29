import numpy as np

class Activation:
    @staticmethod
    def linear(x): 
        return x

    @staticmethod
    def d_linear(x): 
        return np.ones_like(x)

    @staticmethod
    def relu(x): 
        return np.maximum(0, x)
    
    @staticmethod
    def d_relu(x): 
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x): 
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def d_sigmoid(x):
        sig = Activation.sigmoid(x)
        return sig * (1 - sig)
    
    @staticmethod
    def tanh(x): 
        return np.tanh(x)
    
    @staticmethod
    def d_tanh(x): 
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def d_softmax(x):
        s = Activation.softmax(x) 
        batch_size, n = s.shape
        jacobian = np.zeros((batch_size, n, n))  

        for b in range(batch_size):
            s_b = s[b, :].reshape(-1, 1) 
            jacobian[b] = np.diagflat(s_b) - np.dot(s_b, s_b.T)  

        return jacobian

    @staticmethod
    def leaky_relu(x, alpha=0.01): 
        return np.maximum(alpha * x, x)
    
    @staticmethod
    def d_leaky_relu(x, alpha=0.01): 
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx
    
    @staticmethod
    def elu(x, alpha=1.0): 
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def d_elu(x, alpha=1.0): 
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    @staticmethod
    def swish(x, beta=1.0): 
        return x * Activation.sigmoid(beta * x)
    
    @staticmethod
    def d_swish(x, beta=1.0): 
        sigmoid_val = Activation.sigmoid(beta * x)
        return beta * sigmoid_val + x * beta * sigmoid_val * (1 - sigmoid_val)
