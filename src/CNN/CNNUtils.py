import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def preprocess_cifar10_data(x_train, y_train, x_test, y_test, validation_split=0.2):
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    if y_train.ndim > 1:
        y_train = y_train.flatten()
    if y_test.ndim > 1:
        y_test = y_test.flatten()
    
    n_train = len(x_train)
    n_val = int(n_train * validation_split)
    
    indices = np.random.permutation(n_train)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def plot_training_history(history, title="Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['loss'], label='Training Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['accuracy'], label='Training Accuracy', color='blue')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, class_names=None):
    if class_names:
        target_names = class_names
    else:
        target_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("Classification Report:")
    print("=" * 60)
    print(report)


def visualize_sample_predictions(x_test, y_true, y_pred, class_names=None, n_samples=10):
    indices = np.random.choice(len(x_test), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(x_test[idx])
        axes[i].axis('off')
        
        true_label = class_names[y_true[idx]] if class_names else f"Class {y_true[idx]}"
        pred_label = class_names[y_pred[idx]] if class_names else f"Class {y_pred[idx]}"
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        title = f"True: {true_label}\nPred: {pred_label}"
        axes[i].set_title(title, color=color, fontsize=10)
    
    plt.suptitle("Sample Predictions (Green=Correct, Red=Incorrect)")
    plt.tight_layout()
    plt.show()


def compare_model_performances(results_dict, metric='f1_macro'):
    models = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Model Performance Comparison ({metric.replace("_", " ").title()})')
    plt.xlabel('Model Configuration')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_model_weights(model, filepath):
    weights_dict = {}
    
    for i, (layer, layer_type) in enumerate(zip(model.layers, model.layer_types)):
        if layer_type in ['conv2d', 'dense']:
            weights_dict[f'layer_{i}_{layer_type}_weights'] = layer.weights
            weights_dict[f'layer_{i}_{layer_type}_biases'] = layer.biases
    
    np.savez(filepath, **weights_dict)
    print(f"Model weights saved to {filepath}")


def load_model_weights(model, filepath):
    weights_data = np.load(filepath)
    
    layer_count = 0
    for i, (layer, layer_type) in enumerate(zip(model.layers, model.layer_types)):
        if layer_type in ['conv2d', 'dense']:
            weights_key = f'layer_{i}_{layer_type}_weights'
            biases_key = f'layer_{i}_{layer_type}_biases'
            
            if weights_key in weights_data and biases_key in weights_data:
                layer.set_weights(weights_data[weights_key], weights_data[biases_key])
                layer_count += 1
    
    print(f"Loaded weights for {layer_count} layers from {filepath}")


def calculate_model_size(model):
    total_params = 0
    trainable_params = 0
    
    for layer, layer_type in zip(model.layers, model.layer_types):
        if layer_type == 'conv2d':
            layer_params = layer.weights.size + layer.biases.size
            total_params += layer_params
            trainable_params += layer_params
        elif layer_type == 'dense':
            layer_params = layer.weights.size + layer.biases.size
            total_params += layer_params
            trainable_params += layer_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }


def benchmark_inference_time(model, x_test, batch_sizes=[1, 16, 32, 64, 128]):
    import time
    
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(x_test):
            continue
            
        test_data = x_test[:batch_size * 10]
        
        start_time = time.time()
        predictions = model.predict(test_data, batch_size=batch_size)
        end_time = time.time()
        
        total_time = end_time - start_time
        samples_per_second = len(test_data) / total_time
        
        results[batch_size] = {
            'total_time': total_time,
            'samples_per_second': samples_per_second,
            'time_per_sample': total_time / len(test_data)
        }
    
    batch_sizes_plot = list(results.keys())
    throughput = [results[bs]['samples_per_second'] for bs in batch_sizes_plot]
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes_plot, throughput, marker='o', linewidth=2, markersize=8)
    plt.title('Inference Throughput vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Samples per Second')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]