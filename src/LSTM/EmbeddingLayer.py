from RNN.BaseLayer import BaseLayer
import numpy as np

class EmbeddingLayer(BaseLayer):
    def __init__(self, vocab_size, embedding_dim, weights=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        if weights is not None:
            self.weights['embedding'] = weights
        else:
            # Xavier initialization
            self.weights['embedding'] = np.random.uniform(
                -np.sqrt(6.0 / (vocab_size + embedding_dim)),
                np.sqrt(6.0 / (vocab_size + embedding_dim)),
                (vocab_size, embedding_dim)
            )
    
    def forward(self, x, training=False):
        batch_size, seq_len = x.shape
        embedded = np.zeros((batch_size, seq_len, self.embedding_dim))
        
        for i in range(batch_size):
            for j in range(seq_len):
                embedded[i, j] = self.weights['embedding'][x[i, j]]
        
        # Cache for backward pass
        self.cache['input_indices'] = x
        self.cache['batch_size'] = batch_size
        self.cache['seq_len'] = seq_len
        
        return embedded
    
    def backward(self, grad_output):
        x = self.cache['input_indices']
        batch_size, seq_len = x.shape
        
        grad_embedding = np.zeros_like(self.weights['embedding'])
        
        for i in range(batch_size):
            for j in range(seq_len):
                idx = x[i, j]
                grad_embedding[idx] += grad_output[i, j]
        
        self.gradients = {'embedding': grad_embedding}
        
        return None