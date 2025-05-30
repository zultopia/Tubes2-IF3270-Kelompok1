import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from CNN.Conv2DLayer import Conv2DLayer
from CNN.PoolingLayers import MaxPooling2D, AveragePooling2D
from CNN.FlattenLayer import FlattenLayer
from CNN.DenseLayer import DenseLayer


class CNNFromScratch:
    def __init__(self, learning_rate=0.001):
        self.layers = []
        self.layer_types = []
        self.compiled = False
        self.learning_rate = learning_rate
        self.training_history = {'loss': [], 'accuracy': []}
    
    def add_layer(self, layer, layer_type):
        self.layers.append(layer)
        self.layer_types.append(layer_type)
    
    def load_from_keras(self, keras_model):
        self.layers = []
        self.layer_types = []
        current_shape = (32, 32, 3)
        
        print("Loading Keras model to from-scratch implementation...")
        
        for i, keras_layer in enumerate(keras_model.layers):
            if isinstance(keras_layer, tf.keras.layers.InputLayer):
                continue
                
            elif isinstance(keras_layer, tf.keras.layers.Conv2D):
                config = keras_layer.get_config()
                layer = Conv2DLayer(
                    input_shape=current_shape,
                    filters=config['filters'],
                    kernel_size=tuple(config['kernel_size']),
                    strides=tuple(config['strides']),
                    padding=config['padding'],
                    activation=config['activation']
                )
                weights = keras_layer.get_weights()
                if len(weights) >= 2:
                    layer.set_weights(weights[0], weights[1])
                
                self.add_layer(layer, 'conv2d')
                current_shape = layer.output_shape
                
            elif isinstance(keras_layer, tf.keras.layers.MaxPooling2D):
                config = keras_layer.get_config()
                layer = MaxPooling2D(
                    pool_size=tuple(config['pool_size']),
                    strides=tuple(config['strides']) if config['strides'] else tuple(config['pool_size']),
                    padding=config['padding']
                )
                self.add_layer(layer, 'maxpool')
                current_shape = layer._calculate_output_shape(current_shape)
                
            elif isinstance(keras_layer, tf.keras.layers.AveragePooling2D):
                config = keras_layer.get_config()
                layer = AveragePooling2D(
                    pool_size=tuple(config['pool_size']),
                    strides=tuple(config['strides']) if config['strides'] else tuple(config['pool_size']),
                    padding=config['padding']
                )
                self.add_layer(layer, 'avgpool')
                current_shape = layer._calculate_output_shape(current_shape)
                
            elif isinstance(keras_layer, tf.keras.layers.Flatten):
                layer = FlattenLayer()
                self.add_layer(layer, 'flatten')
                current_shape = (np.prod(current_shape),)
                
            elif isinstance(keras_layer, tf.keras.layers.Dense):
                config = keras_layer.get_config()
                layer = DenseLayer(
                    input_size=current_shape[0],
                    output_size=config['units'],
                    activation=config['activation']
                )
                weights = keras_layer.get_weights()
                if len(weights) >= 2:
                    layer.set_weights(weights[0], weights[1])
                
                self.add_layer(layer, 'dense')
                current_shape = (config['units'],)
                
            elif isinstance(keras_layer, tf.keras.layers.Dropout):
                continue 
        
        self.compiled = True
        print(f"Model loaded successfully with {len(self.layers)} layers")
    
    def forward(self, x):
        current_output = x
        
        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer_type == 'dense' and current_output.ndim > 2:
                batch_size = current_output.shape[0]
                current_output = current_output.reshape(batch_size, -1)
            current_output = layer.forward(current_output)
        
        return current_output
    
    def backward(self, grad_output):
        current_grad = grad_output
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                current_grad = layer.backward(current_grad)
    
    def update_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'gradients') and hasattr(layer, 'weights'):
                if hasattr(layer.gradients, 'get'):
                    if 'kernel' in layer.gradients and layer.gradients['kernel'] is not None:
                        layer.weights['kernel'] -= self.learning_rate * layer.gradients['kernel']
                    if 'bias' in layer.gradients and layer.gradients['bias'] is not None:
                        layer.biases['bias'] -= self.learning_rate * layer.gradients['bias']
    
    def predict(self, x, batch_size=32):
        if not self.compiled:
            raise ValueError("Model must be loaded before prediction")
        
        n_samples = len(x)
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_x = x[i:end_idx]
            batch_probs = self.forward(batch_x)
            batch_preds = np.argmax(batch_probs, axis=1)
            predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    def predict_proba(self, x, batch_size=32):
        if not self.compiled:
            raise ValueError("Model must be loaded before prediction")
        
        n_samples = len(x)
        all_probs = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_x = x[i:end_idx]
            batch_probs = self.forward(batch_x)
            all_probs.append(batch_probs)
        
        return np.concatenate(all_probs, axis=0)
    
    def evaluate(self, x, y, batch_size=32):
        predictions = self.predict(x, batch_size)
        accuracy = np.mean(predictions == y)
        f1_macro = f1_score(y, predictions, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'predictions': predictions
        }
    
    def sparse_categorical_crossentropy(self, y_true, y_pred):
        batch_size = y_pred.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        log_likelihood = -np.log(y_pred_clipped[range(batch_size), y_true])
        return np.mean(log_likelihood)
    
    def sparse_categorical_crossentropy_gradient(self, y_true, y_pred):
        batch_size = y_pred.shape[0]
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[range(batch_size), y_true] = 1
        return (y_pred - y_true_one_hot) / batch_size
    
    def train_step(self, x_batch, y_batch):
        predictions = self.forward(x_batch)
        
        loss = self.sparse_categorical_crossentropy(y_batch, predictions)
        
        grad_loss = self.sparse_categorical_crossentropy_gradient(y_batch, predictions)
        self.backward(grad_loss)
        
        self.update_weights()
        
        pred_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(pred_classes == y_batch)
        
        return loss, accuracy
    
    def fit(self, x_train, y_train, epochs=10, batch_size=32, 
            validation_data=None, verbose=1):
        n_samples = len(x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                loss, accuracy = self.train_step(x_batch, y_batch)
                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
            
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(avg_accuracy)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f}", end="")
                
                if validation_data:
                    x_val, y_val = validation_data
                    val_pred = self.predict_proba(x_val, batch_size=batch_size)
                    val_loss = self.sparse_categorical_crossentropy(y_val, val_pred)
                    val_pred_classes = np.argmax(val_pred, axis=1)
                    val_accuracy = np.mean(val_pred_classes == y_val)
                    print(f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
                else:
                    print()
        
        return self.training_history
    
    def compare_with_keras(self, keras_model, x_test, y_test, batch_size=32):
        print("Comparing forward propagation with Keras model...")
        
        keras_probs = keras_model.predict(x_test, batch_size=batch_size, verbose=0)
        scratch_probs = self.predict_proba(x_test, batch_size=batch_size)
        
        prob_diff = np.abs(keras_probs - scratch_probs)
        max_diff = np.max(prob_diff)
        mean_diff = np.mean(prob_diff)
        
        keras_preds = np.argmax(keras_probs, axis=1)
        scratch_preds = np.argmax(scratch_probs, axis=1)
        
        keras_f1 = f1_score(y_test, keras_preds, average='macro')
        scratch_f1 = f1_score(y_test, scratch_preds, average='macro')
        
        predictions_match = np.mean(keras_preds == scratch_preds)
        
        print(f"Max probability difference: {max_diff:.8f}")
        print(f"Mean probability difference: {mean_diff:.8f}")
        print(f"Prediction match rate: {predictions_match*100:.2f}%")
        print(f"Keras F1-macro: {keras_f1:.6f}")
        print(f"From Scratch F1-macro: {scratch_f1:.6f}")
        print(f"F1 difference: {abs(keras_f1 - scratch_f1):.6f}")
        
        return {
            'max_probability_difference': max_diff,
            'mean_probability_difference': mean_diff,
            'predictions_match_percentage': predictions_match * 100,
            'keras_f1_macro': keras_f1,
            'scratch_f1_macro': scratch_f1,
            'f1_difference': abs(keras_f1 - scratch_f1)
        }
    
    def summary(self):
        if not self.compiled:
            print("Model not loaded/compiled")
            return
        
        print("=" * 60)
        print("CNN From Scratch Model Summary")
        print("=" * 60)
        print(f"{'Layer Type':<15} {'Output Shape':<20} {'Parameters'}")
        print("-" * 60)
        
        total_params = 0
        
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            params = 0
            
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                if 'kernel' in layer.weights and 'bias' in layer.biases:
                    params = layer.weights['kernel'].size + layer.biases['bias'].size
            
            if layer_type == 'conv2d':
                output_shape = layer.output_shape
            elif layer_type in ['maxpool', 'avgpool']:
                output_shape = getattr(layer, 'output_shape', 'N/A')
            elif layer_type == 'dense':
                output_shape = (layer.output_size,)
            else:
                output_shape = 'N/A'
            
            print(f"{layer_type:<15} {str(output_shape):<20} {params:,}")
            total_params += params
        
        print("-" * 60)
        print(f"Total parameters: {total_params:,}")
        print("=" * 60)