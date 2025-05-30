# 🧠 Deep Learning from Scratch: CNN & RNN Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-orange.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Tugas Besar #2 IF3270 Pembelajaran Mesin**  
> *Implementasi Convolutional Neural Network dan Recurrent Neural Network dari Awal*

## 📝 Deskripsi

Proyek ini mengimplementasikan tiga arsitektur deep learning utama **dari awal (from scratch)** menggunakan hanya NumPy sebagai library matematika dasar:

- 🖼️ **Convolutional Neural Network (CNN)** untuk klasifikasi gambar CIFAR-10
- 🔄 **Simple Recurrent Neural Network (RNN)** untuk klasifikasi sentimen teks
- 🧮 **Long Short-Term Memory (LSTM)** untuk klasifikasi sentimen teks

Implementasi mencakup forward propagation, backward propagation, dan training loop yang complete, kemudian dibandingkan dengan implementasi Keras untuk validasi akurasi.

## 👥 Disusun Oleh Kelompok 1

| Nama | NIM | Kontribusi |
|------|-----|------------|
| **Marzuli Suhada M** | 13522070 | LSTM Implementation & Backward Propagation |
| **Ahmad Mudabbir Arif** | 13522072 | CNN Implementation & Backward Propagation |
| **Naufal Adnan** | 13522116 | Simple RNN Implementation & Backward Propagation |

## 📁 Struktur Proyek

```
TUBES2-ML/
├── 📄 doc/
│   └── Laporan Tubes 2 Kelompok 1.pdf
├── 💻 src/
│   ├── 🖼️ CNN/
│   │   ├── BaseLayer.py
│   │   ├── CNNFromScratch.py
│   │   ├── CNNUtils.py
│   │   ├── Conv2DLayer.py
│   │   ├── DenseLayer.py
│   │   ├── FlattenLayer.py
│   │   └── PoolingLayers.py
│   ├── 📊 dataset/
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── valid.csv
│   ├── 🧮 LSTM/
│   │   ├── BaseLayer.py
│   │   ├── BidirectionalLSTM.py
│   │   ├── DenseLayer.py
│   │   ├── DropoutLayer.py
│   │   ├── EmbeddingLayer.py
│   │   ├── LSTMFromScratch.py
│   │   └── LSTMLayer.py
│   ├── 🔄 RNN/
│   │   ├── BaseLayer.py
│   │   ├── BidirectionalRNNLayer.py
│   │   ├── DenseLayer.py
│   │   ├── DropoutLayer.py
│   │   ├── EmbeddingLayer.py
│   │   ├── RNNFromScratch.py
│   │   └── SimpleRNNLayer.py
|   ├── 🧪 test/
|   │   ├── 📈 output_model/
|   │   │   ├── best_cnn_model.h5
|   │   │   ├── lstm_model.h5
|   │   │   └── rnn_model.h5
│   │   ├── CNN.ipynb
│   │   ├── LSTM.ipynb
│   │   └── simpleRNN.ipynb
│   └── 🛠️ utils/
│       ├── Activation.py
│       └── __init__.py
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn sklearn keras tensorflow
```

### Installation

```bash
https://github.com/zultopia/Tubes2-IF3270-Kelompok1
cd Tubes2-IF3270-Kelompok1
```

### Running the Models

#### 1. CNN untuk Klasifikasi Gambar CIFAR-10
```bash
cd test
jupyter notebook CNN.ipynb
```

#### 2. Simple RNN untuk Klasifikasi Sentimen
```bash
cd test
jupyter notebook simpleRNN.ipynb
```

#### 3. LSTM untuk Klasifikasi Sentimen
```bash
cd test
jupyter notebook LSTM.ipynb
```

## ✨ Fitur Utama

### 🔧 Implementasi From Scratch
- ✅ **Forward Propagation** lengkap untuk semua layer
- ✅ **Backward Propagation** dengan Backpropagation Through Time (BPTT)
- ✅ **Batch Processing** untuk training dan inference yang efisien
- ✅ **Modular Architecture** dengan base classes dan inheritance

### 🧪 Eksperimen Komprehensif

#### CNN Analysis
- 📊 Pengaruh jumlah layer konvolusi (1, 2, 3 layers)
- 🔍 Pengaruh jumlah filter per layer (16-32, 32-64, 64-128)
- 📏 Pengaruh ukuran kernel (3x3, 5x5, 7x7+3x3)
- 🏊 Perbandingan Max Pooling vs Average Pooling

#### RNN & LSTM Analysis
- 📈 Pengaruh jumlah layer (1, 2, 3 layers)
- 🧠 Pengaruh jumlah hidden units (32, 64, 128 cells)
- ↔️ Perbandingan Unidirectional vs Bidirectional processing

### 📊 Validasi dan Benchmarking
- 🎯 **100% Prediction Agreement** dengan Keras pada forward propagation
- 📈 Perbandingan performa menggunakan Macro F1-Score
- 📉 Analisis training/validation loss curves
- 🔍 Comprehensive error analysis

## 📈 Hasil Performa

### CNN (CIFAR-10)
| Konfigurasi | Test Accuracy | Macro F1-Score | Parameters |
|-------------|---------------|----------------|------------|
| 3 Conv Layers | **71.77%** | **0.7143** | 356,810 |
| Large Filters | 70.98% | 0.7065 | 1,125,642 |
| 3x3 Kernels | 69.68% | 0.6965 | 545,098 |

### RNN & LSTM (NusaX-Sentiment)
| Model | Architecture | Macro F1-Score | Parameters |
|-------|--------------|----------------|------------|
| **LSTM (1 Layer)** | Bidirectional | **0.7505** | 890,755 |
| RNN (2 Layers) | Bidirectional | 0.5571 | 816,899 |
| RNN (Bidirectional) | Single Layer | 0.5701 | 825,347 |

## 📊 Dataset

- **CIFAR-10**: 50,000 training + 10,000 test images (32x32 RGB)
  - Split: 40k train, 10k validation, 10k test
- **NusaX-Sentiment**: Indonesian sentiment classification dataset
  - 3 classes: positive, negative, neutral

## 🛠️ Teknologi

- **Core**: Python 3.8+, NumPy
- **Validation**: Keras/TensorFlow
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: Scikit-learn
- **Development**: Jupyter Notebook

## 📚 Key Learnings

### 🏆 Achievements
- ✅ **Perfect Forward Propagation**: 100% agreement dengan Keras
- ✅ **Complete BPTT Implementation**: Full backward propagation untuk RNN/LSTM
- ✅ **Batch Processing**: Efficient training dengan berbagai batch sizes
- ✅ **Hyperparameter Analysis**: Comprehensive experiments dengan 15+ konfigurasi

### 🎓 Insights
- 🧠 **CNN**: Kernel 3x3 optimal untuk images kecil, Max Pooling > Average Pooling
- 🔄 **RNN**: Bidirectional processing memberikan +51% improvement
- 🧮 **LSTM**: Single layer lebih baik dari multi-layer pada dataset kecil
- ⚖️ **Trade-off**: Model complexity vs generalization capability

## 📖 Documentation

Dokumentasi lengkap tersedia di [Laporan Tugas Besar](doc/Laporan%20Tubes%202%20Kelompok%201.pdf) yang mencakup:
- 🔬 Analisis mendalam setiap arsitektur
- 📊 Visualisasi hasil eksperimen
- 🧮 Formulasi matematis implementasi
- 📈 Perbandingan performa detail

## 🔧 Usage Examples

### Menggunakan CNN From Scratch
```python
from src.CNN import CNNFromScratch

# Load model
cnn_model = CNNFromScratch()
cnn_model.load_from_keras('test/output_model/best_cnn_model.h5')

# Prediksi
predictions = cnn_model.predict(x_test, batch_size=32)
f1_score = cnn_model.evaluate(x_test, y_test)
```

### Menggunakan LSTM From Scratch
```python
from src.LSTM import LSTMFromScratch

# Inisialisasi model
lstm_model = LSTMFromScratch(learning_rate=0.001)
lstm_model.add_embedding_layer(vocab_size, embedding_dim)
lstm_model.add_bidirectional_lstm_layer(units=64)
lstm_model.add_dense_layer(units=num_classes, activation='softmax')

# Training
history = lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
```

## 🧪 Reproducing Results

Untuk mereproduksi hasil eksperimen:

1. **CNN Experiments**:
   ```bash
   cd test
   jupyter notebook CNN.ipynb
   # Jalankan semua cell untuk eksperimen lengkap
   ```

2. **RNN Experiments**:
   ```bash
   cd test
   jupyter notebook simpleRNN.ipynb
   # Eksperimen hyperparameter RNN
   ```

3. **LSTM Experiments**:
   ```bash
   cd test
   jupyter notebook LSTM.ipynb
   # Analisis performa LSTM
   ```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Issues & Troubleshooting

### Common Issues
- **Memory Error**: Reduce batch size dalam training
- **Slow Training**: Gunakan batch processing yang lebih kecil
- **Gradient Explosion**: Implement gradient clipping

### Contact
Jika menemukan bug atau memiliki pertanyaan, silakan buat issue atau hubungi tim pengembang.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- **Institut Teknologi Bandung** - IF3270 Pembelajaran Mesin
- **Dive into Deep Learning** - Mathematical foundations
- **Keras Documentation** - Implementation reference
- **NumPy Community** - Core computational library
- **NusaX Team** - Indonesian sentiment dataset

## 📚 References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
- LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition.
- Zhang, A., et al. (2021). Dive into Deep Learning. Available at: https://d2l.ai/

---

<div align="center">

**🎓 Tugas Besar IF3270 Pembelajaran Mesin 2024/2025**

Made with ❤️ by Kelompok 1

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/zultopia/Tubes2-IF3270-Kelompok1)
[![Report](https://img.shields.io/badge/📄-Laporan-blue.svg)](doc/Laporan%20Tubes%202%20Kelompok%201.pdf)

</div>