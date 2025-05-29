from RNN.EmbeddingLayer import EmbeddingLayer
from LSTM.LSTMLayer import LSTMLayer
from LSTM.BidirectionalLSTMLayer import BidirectionalLSTMLayer
from RNN.DropoutLayer import DropoutLayer
from RNN.DenseLayer import DenseLayer
import numpy as np
import json
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

class LSTMFromScratch:
    def __init__(self, learning_rate=0.001):
        self.layers = []
        self.learning_rate = learning_rate
        self.encoder = None
        
    def add_embedding_layer(self, vocab_size, embedding_dim, weights=None):
        """Add embedding layer to the model"""
        layer = EmbeddingLayer(vocab_size, embedding_dim, weights)
        self.layers.append(layer)
        
    def add_lstm_layer(self, units, return_sequences=True, weights=None):
        """Add LSTM layer to the model"""
        layer = LSTMLayer(units, return_sequences, weights)
        self.layers.append(layer)
        
    def add_bidirectional_lstm_layer(self, units, return_sequences=True, weights=None):
        """Add bidirectional LSTM layer to the model"""
        layer = BidirectionalLSTMLayer(units, return_sequences, weights)
        self.layers.append(layer)
        
    def add_dropout_layer(self, rate):
        """Add dropout layer to the model"""
        layer = DropoutLayer(rate)
        self.layers.append(layer)
        
    def add_dense_layer(self, units, activation='softmax', weights=None):
        """Add dense layer to the model"""
        layer = DenseLayer(units, activation, weights)
        self.layers.append(layer)
    
    def forward(self, x, training=False):
        """Forward propagation through all layers"""
        current_output = x
        
        for layer in self.layers:
            current_output = layer.forward(current_output, training)
        
        return current_output
    
    def backward(self, grad_output):
        """Backward propagation through all layers"""
        current_grad = grad_output
        
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                current_grad = layer.backward(current_grad)
    
    def categorical_crossentropy_loss(self, y_true, y_pred):
        """Calculate categorical crossentropy loss"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def categorical_crossentropy_gradient(self, y_true, y_pred):
        """Calculate gradient of categorical crossentropy loss"""
        return (y_pred - y_true) / y_true.shape[0]
    
    def update_weights(self):
        """Update weights using calculated gradients"""
        for layer in self.layers:
            if hasattr(layer, 'gradients'):
                for param_name, gradient in layer.gradients.items():
                    if param_name in layer.weights:
                        layer.weights[param_name] -= self.learning_rate * gradient
                    elif param_name in layer.biases:
                        layer.biases[param_name] -= self.learning_rate * gradient
    
    def train_step(self, x_batch, y_batch):
        """Perform one training step"""
        # Forward pass
        predictions = self.forward(x_batch, training=True)
        
        # Calculate loss
        loss = self.categorical_crossentropy_loss(y_batch, predictions)
        
        # Backward pass
        grad_output = self.categorical_crossentropy_gradient(y_batch, predictions)
        self.backward(grad_output)
        
        # Update weights
        self.update_weights()
        
        return loss, predictions
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None, verbose=1):
        """Train the model"""
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Handle label encoding
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
        """Make predictions on input data"""
        num_samples = len(x)
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch_x = x[i:i+batch_size]
            batch_pred = self.forward(batch_x, training=False)
            predictions.append(batch_pred)
        
        return np.vstack(predictions)
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate model performance"""
        predictions = self.predict(X_test, batch_size=batch_size)
        pred_classes = np.argmax(predictions, axis=1)
        
        if self.encoder is not None:
            y_test_encoded = self.encoder.transform(y_test.reshape(-1, 1))
            true_classes = np.argmax(y_test_encoded, axis=1)
        else:
            if y_test.ndim > 1:
                true_classes = np.argmax(y_test, axis=1)
            else:
                true_classes = y_test
        
        f1 = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
        return f1, pred_classes, true_classes
    
    def save_model(self, filepath):
        """Save model to file"""
        model_data = {
            'layers': [],
            'learning_rate': self.learning_rate,
            'encoder': None
        }
        
        # Save encoder if exists
        if self.encoder is not None:
            model_data['encoder'] = {
                'categories': [cat.tolist() for cat in self.encoder.categories_],
                'feature_names_in': getattr(self.encoder, 'feature_names_in_', None)
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
            elif isinstance(layer, (LSTMLayer, BidirectionalLSTMLayer)):
                layer_data['units'] = layer.units
                layer_data['return_sequences'] = layer.return_sequences
            elif isinstance(layer, DropoutLayer):
                layer_data['rate'] = layer.rate
            elif isinstance(layer, DenseLayer):
                layer_data['units'] = layer.units
                layer_data['activation'] = layer.activation_name
            
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        model = cls(learning_rate=model_data.get('learning_rate', 0.001))
        
        # Restore encoder if exists
        if model_data.get('encoder') is not None:
            encoder_data = model_data['encoder']
            model.encoder = OneHotEncoder(sparse_output=False)
            model.encoder.categories_ = [np.array(cat) for cat in encoder_data['categories']]
            if encoder_data.get('feature_names_in_') is not None:
                model.encoder.feature_names_in_ = np.array(encoder_data['feature_names_in_'])
        
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
            elif layer_type == 'LSTMLayer':
                layer_weights = [
                    weights['kernel'],
                    weights['recurrent_kernel'],
                    biases['bias']
                ]
                layer = LSTMLayer(
                    layer_data['units'],
                    layer_data['return_sequences'],
                    layer_weights
                )
            elif layer_type == 'BidirectionalLSTMLayer':
                layer_weights = [
                    weights['forward_kernel'],
                    weights['forward_recurrent'],
                    biases['forward_bias'],
                    weights['backward_kernel'],
                    weights['backward_recurrent'],
                    biases['backward_bias']
                ]
                layer = BidirectionalLSTMLayer(
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
        
        print(f"Model loaded from {filepath}")
        return model
    
    def load_keras_model(self, keras_model):
        """Load weights from a Keras model"""
        print("\nLoading Keras model into LSTMFromScratch...")
        
        self.layers = []

        for keras_layer in keras_model.layers:
            if isinstance(keras_layer, tf.keras.layers.Embedding):
                weights = keras_layer.get_weights()[0]
                self.add_embedding_layer(
                    vocab_size=weights.shape[0], 
                    embedding_dim=weights.shape[1], 
                    weights=weights
                )
                print(f"Added EmbeddingLayer: vocab_size={weights.shape[0]}, embedding_dim={weights.shape[1]}")
                
            elif isinstance(keras_layer, tf.keras.layers.LSTM):
                weights = keras_layer.get_weights()
                return_seq = keras_layer.return_sequences
                self.add_lstm_layer(
                    units=keras_layer.units,
                    return_sequences=return_seq,
                    weights=weights
                )
                print(f"Added LSTMLayer: units={keras_layer.units}, return_sequences={return_seq}")
                
            elif isinstance(keras_layer, tf.keras.layers.Bidirectional):
                # Handle bidirectional LSTM
                if isinstance(keras_layer.layer, tf.keras.layers.LSTM):
                    forward_weights = keras_layer.forward_layer.get_weights()
                    backward_weights = keras_layer.backward_layer.get_weights()
                    all_weights = forward_weights + backward_weights
                    
                    return_seq = keras_layer.forward_layer.return_sequences
                    total_units = keras_layer.forward_layer.units * 2
                    
                    self.add_bidirectional_lstm_layer(
                        units=total_units, 
                        return_sequences=return_seq,
                        weights=all_weights
                    )
                    print(f"Added BidirectionalLSTMLayer: units={total_units}, return_sequences={return_seq}")
                
            elif isinstance(keras_layer, tf.keras.layers.Dropout):
                self.add_dropout_layer(rate=keras_layer.rate)
                print(f"Added DropoutLayer: rate={keras_layer.rate}")
                
            elif isinstance(keras_layer, tf.keras.layers.Dense):
                weights = keras_layer.get_weights()
                activation_name = keras_layer.activation.__name__
                self.add_dense_layer(
                    units=keras_layer.units,
                    activation=activation_name,
                    weights=weights
                )
                print(f"Added DenseLayer: units={keras_layer.units}, activation={activation_name}")
        
        print(f"\nModel loaded successfully!")
        print(f"Total layers: {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1}: {layer.__class__.__name__}")
    
    def compare_with_keras(self, keras_model, X_test, y_test, batch_size=32):
        """Compare predictions with Keras model"""
        print("\n=== Comparing with Keras Model ===")
        
        # Get Keras predictions
        keras_pred = keras_model.predict(X_test, batch_size=batch_size, verbose=0)
        keras_pred_classes = np.argmax(keras_pred, axis=1)
        
        # Get from-scratch predictions
        scratch_pred = self.predict(X_test, batch_size=batch_size)
        scratch_pred_classes = np.argmax(scratch_pred, axis=1)
        
        # Calculate F1 scores
        keras_f1 = f1_score(y_test, keras_pred_classes, average='macro')
        scratch_f1 = f1_score(y_test, scratch_pred_classes, average='macro')
        
        # Calculate prediction similarity
        prediction_match = np.mean(keras_pred_classes == scratch_pred_classes)
        
        # Calculate numerical similarity of outputs
        output_similarity = np.mean(np.abs(keras_pred - scratch_pred))
        
        print(f"Keras Model F1-score: {keras_f1:.4f}")
        print(f"From-scratch Model F1-score: {scratch_f1:.4f}")
        print(f"F1-score difference: {abs(keras_f1 - scratch_f1):.4f}")
        print(f"Prediction match rate: {prediction_match:.4f}")
        print(f"Average output difference: {output_similarity:.6f}")
        
        return {
            'keras_f1': keras_f1,
            'scratch_f1': scratch_f1,
            'f1_difference': abs(keras_f1 - scratch_f1),
            'prediction_match': prediction_match,
            'output_similarity': output_similarity,
            'keras_predictions': keras_pred_classes,
            'scratch_predictions': scratch_pred_classes
        }
    
    def plot_history(self, history):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss', marker='o')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss', marker='s')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # F1 Score
        plt.subplot(1, 2, 2)
        plt.plot(history['f1'], label='Training F1 Score', marker='^')
        if 'val_f1' in history:
            plt.plot(history['val_f1'], label='Validation F1 Score', marker='d')
        plt.title('Model F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self):
        """Get model architecture summary"""
        print("\n=== Model Architecture Summary ===")
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_params = 0
            layer_info = f"Layer {i+1}: {layer.__class__.__name__}"
            
            if hasattr(layer, 'weights'):
                for name, weight in layer.weights.items():
                    layer_params += weight.size
                    layer_info += f"\n  {name}: {weight.shape}"
            
            if hasattr(layer, 'biases'):
                for name, bias in layer.biases.items():
                    layer_params += bias.size
                    layer_info += f"\n  {name}: {bias.shape}"
            
            # Add layer-specific info
            if isinstance(layer, EmbeddingLayer):
                layer_info += f"\n  vocab_size: {layer.vocab_size}, embedding_dim: {layer.embedding_dim}"
            elif isinstance(layer, (LSTMLayer, BidirectionalLSTMLayer)):
                layer_info += f"\n  units: {layer.units}, return_sequences: {layer.return_sequences}"
            elif isinstance(layer, DropoutLayer):
                layer_info += f"\n  rate: {layer.rate}"
            elif isinstance(layer, DenseLayer):
                layer_info += f"\n  units: {layer.units}, activation: {layer.activation_name}"
            
            layer_info += f"\n  Parameters: {layer_params:,}"
            print(layer_info)
            print("-" * 50)
            
            total_params += layer_params
        
        print(f"Total parameters: {total_params:,}")
        return total_params