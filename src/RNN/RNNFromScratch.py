from RNN.EmbeddingLayer import EmbeddingLayer
from RNN.SimpleRNNLayer import SimpleRNNLayer
from RNN.BidirectionalRNNLayer import BidirectionalRNNLayer
from RNN.DropoutLayer import DropoutLayer
from RNN.DenseLayer import DenseLayer
import numpy as np
import json
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

class RNNFromScratch:
    def __init__(self, learning_rate=0.001):
        self.layers = []
        self.learning_rate = learning_rate
        
    def add_embedding_layer(self, vocab_size, embedding_dim, weights=None):
        layer = EmbeddingLayer(vocab_size, embedding_dim, weights)
        self.layers.append(layer)
        
    def add_simple_rnn_layer(self, units, return_sequences=True, weights=None):
        layer = SimpleRNNLayer(units, return_sequences, weights)
        self.layers.append(layer)
        
    def add_bidirectional_rnn_layer(self, units, return_sequences=True, weights=None):
        layer = BidirectionalRNNLayer(units, return_sequences, weights)
        self.layers.append(layer)
        
    def add_dropout_layer(self, rate):
        layer = DropoutLayer(rate)
        self.layers.append(layer)
        
    def add_dense_layer(self, units, activation='softmax', weights=None):
        layer = DenseLayer(units, activation, weights)
        self.layers.append(layer)
    
    def forward(self, x, training=False):
        current_output = x
        
        for layer in self.layers:
            current_output = layer.forward(current_output, training)
        
        return current_output
    
    def backward(self, grad_output):
        current_grad = grad_output
        
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                current_grad = layer.backward(current_grad)
    
    def categorical_crossentropy_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def categorical_crossentropy_gradient(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]
    
    def update_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'gradients'):
                for param_name, gradient in layer.gradients.items():
                    if param_name in layer.weights:
                        layer.weights[param_name] -= self.learning_rate * gradient
                    elif param_name in layer.biases:
                        layer.biases[param_name] -= self.learning_rate * gradient
    
    def train_step(self, x_batch, y_batch):
        # Forward pass
        predictions = self.forward(x_batch, training=True)
        
        # loss
        loss = self.categorical_crossentropy_loss(y_batch, predictions)
        
        # Backward pass
        grad_output = self.categorical_crossentropy_gradient(y_batch, predictions)
        self.backward(grad_output)
        
        # Update weights
        self.update_weights()
        
        return loss, predictions
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None, verbose=1):
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        if y_train.ndim == 1 or y_train.shape[1] == 1:
            self.encoder = OneHotEncoder(sparse_output=False)
            y_train = self.encoder.fit_transform(y_train.reshape(-1, 1))
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val = self.encoder.transform(y_val.reshape(-1, 1))
                validation_data = (X_val, y_val)
        else:
            self.encoder = None
        
        history = {'loss': [], 'val_loss': [], 'f1': [], 'val_f1': []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            all_preds = []
            all_true = []
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                x_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Train step
                batch_loss, batch_preds = self.train_step(x_batch, y_batch)
                epoch_loss += batch_loss
                
                all_preds.extend(np.argmax(batch_preds, axis=1))
                all_true.extend(np.argmax(y_batch, axis=1))
            
            avg_loss = epoch_loss / n_batches
            f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
            
            history['loss'].append(avg_loss)
            history['f1'].append(f1)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_preds = self.predict(X_val, batch_size=batch_size)
                val_loss = self.categorical_crossentropy_loss(y_val, val_preds)
                val_f1 = f1_score(np.argmax(y_val, axis=1), np.argmax(val_preds, axis=1), 
                                average='macro', zero_division=0)
                
                history['val_loss'].append(val_loss)
                history['val_f1'].append(val_f1)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {avg_loss:.4f} - f1: {f1:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_f1: {val_f1:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - f1: {f1:.4f}")
        
        return history
    
    def predict(self, x, batch_size=32):
        num_samples = len(x)
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch_x = x[i:i+batch_size]
            batch_pred = self.forward(batch_x, training=False)
            predictions.append(batch_pred)
        
        return np.vstack(predictions)
    
    def save_model(self, filepath):
        model_data = {
            'layers': [],
            'learning_rate': self.learning_rate
        }
        
        for i, layer in enumerate(self.layers):
            layer_data = {
                'type': layer.__class__.__name__,
                'weights': {k: v.tolist() for k, v in layer.weights.items()},
                'biases': {k: v.tolist() for k, v in layer.biases.items()}
            }
            
            # Add layer-specific parameters
            if isinstance(layer, EmbeddingLayer):
                layer_data['vocab_size'] = layer.vocab_size
                layer_data['embedding_dim'] = layer.embedding_dim
            elif isinstance(layer, (SimpleRNNLayer, BidirectionalRNNLayer)):
                layer_data['units'] = layer.units
                layer_data['return_sequences'] = layer.return_sequences
            elif isinstance(layer, DropoutLayer):
                layer_data['rate'] = layer.rate
            elif isinstance(layer, DenseLayer):
                layer_data['units'] = layer.units
                layer_data['activation'] = layer.activation
            
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        model = cls(learning_rate=model_data.get('learning_rate', 0.001))
        
        for layer_data in model_data['layers']:
            layer_type = layer_data['type']
            weights = {k: np.array(v) for k, v in layer_data['weights'].items()}
            biases = {k: np.array(v) for k, v in layer_data['biases'].items()}
            
            if layer_type == 'EmbeddingLayer':
                layer = EmbeddingLayer(
                    layer_data['vocab_size'], 
                    layer_data['embedding_dim'],
                    weights.get('embedding')
                )
            elif layer_type == 'SimpleRNNLayer':
                layer_weights = [
                    weights['kernel'],
                    weights['recurrent_kernel'],
                    biases['bias']
                ]
                layer = SimpleRNNLayer(
                    layer_data['units'],
                    layer_data['return_sequences'],
                    layer_weights
                )
            elif layer_type == 'BidirectionalRNNLayer':
                layer_weights = [
                    weights['forward_kernel'],
                    weights['forward_recurrent'],
                    biases['forward_bias'],
                    weights['backward_kernel'],
                    weights['backward_recurrent'],
                    biases['backward_bias']
                ]
                layer = BidirectionalRNNLayer(
                    layer_data['units'],
                    layer_data['return_sequences'],
                    layer_weights
                )
            elif layer_type == 'DropoutLayer':
                layer = DropoutLayer(layer_data['rate'])
            elif layer_type == 'DenseLayer':
                layer_weights = [weights['kernel'], biases['bias']]
                layer = DenseLayer(
                    layer_data['units'],
                    layer_data['activation'],
                    layer_weights
                )
            
            layer.weights = weights
            layer.biases = biases
            model.layers.append(layer)
        
        return model
    
    def load_keras_model(self, keras_model):
        print("\nLoading Keras model into RNNFromScratch...")
        
        self.layers = []

        for keras_layer in keras_model.layers:
            if isinstance(keras_layer, tf.keras.layers.Embedding):
                weights = keras_layer.get_weights()[0]
                self.add_embedding_layer(
                    vocab_size=weights.shape[0], 
                    embedding_dim=weights.shape[1], 
                    weights=weights
                )
                
            elif isinstance(keras_layer, tf.keras.layers.SimpleRNN):
                weights = keras_layer.get_weights()
                return_seq = keras_layer.return_sequences
                self.add_simple_rnn_layer(
                    units=keras_layer.units,
                    return_sequences=return_seq,
                    weights=weights
                )
                
            elif isinstance(keras_layer, tf.keras.layers.Bidirectional):
                forward_weights = keras_layer.forward_layer.get_weights()
                backward_weights = keras_layer.backward_layer.get_weights()
                all_weights = forward_weights + backward_weights
                
                return_seq = keras_layer.forward_layer.return_sequences
                self.add_bidirectional_rnn_layer(
                    units=keras_layer.forward_layer.units * 2, 
                    return_sequences=return_seq,
                    weights=all_weights
                )
                
            elif isinstance(keras_layer, tf.keras.layers.Dropout):
                self.add_dropout_layer(rate=keras_layer.rate)
                
            elif isinstance(keras_layer, tf.keras.layers.Dense):
                weights = keras_layer.get_weights()
                self.add_dense_layer(
                    units=keras_layer.units,
                    activation=keras_layer.activation.__name__,
                    weights=weights
                )
        
        print(f"\nModel loaded successfully. \n{len(self.layers)} layers:")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1}: {layer.__class__.__name__}")
    
    def plot_history(self, history):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # F1 Score
        plt.subplot(1, 2, 2)
        plt.plot(history['f1'], label='Training F1 Score')
        if 'val_f1' in history:
            plt.plot(history['val_f1'], label='Validation F1 Score')
        plt.title('F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()