# Retinal OCT Disease Classification using MobileNetV3Large

This project implements a high-performance Deep Learning model to classify retinal diseases from **Optical Coherence Tomography (OCT)** images. Using **Transfer Learning** with the **MobileNetV3Large** architecture, the model achieves a state-of-the-art accuracy of **97%**.

## 📄 Abstract
Retinal diseases such as CNV and DME are leading causes of blindness. Early detection through OCT imaging is crucial. This project automates the diagnostic process by classifying OCT scans into four distinct categories with high precision and F1-score, making it a reliable tool for clinical decision support.

## 📊 Dataset Metadata
The model was trained on the [Labeled Optical Coherence Tomography (OCT)](https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct) dataset.

* **Total Images:** ~109,000
* **Categories:** 4 (CNV, DME, DRUSEN, NORMAL)
* **Data Split:**
    * **Training:** 76,515 images
    * **Validation:** 21,861 images
    * **Testing:** 10,933 images
* **Image Resolution:** Resized to 224x224 pixels for model compatibility.

## 🏗️ Technical Architecture
The core of the system is based on **MobileNetV3Large**, chosen for its optimal balance between computational efficiency and feature extraction capability.

* **Pre-trained Weights:** ImageNet
* **Optimization:** Adam Optimizer ($LR = 10^{-4}$)
* **Loss Function:** Categorical Crossentropy
* **Input Shape:** (224, 224, 3)
* **Framework:** TensorFlow / Keras
* **Inference Style:** Categorical classification with Softmax activation.

## 🚀 Model Performance
After 15 epochs of training, the model demonstrated excellent convergence and generalization on unseen test data.

### 📈 Test Set Metrics
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **97.09%** |
| **F1-Score (Weighted)** | **97.12%** |
| **Loss** | **0.1260** |

### 📋 Classification Report
Detailed performance per class:
* **Normal:** Precision: 0.97 | Recall: 0.98 | F1: 0.98
* **CNV:** Precision: 0.98 | Recall: 0.99 | F1: 0.99
* **DME:** Precision: 0.97 | Recall: 0.94 | F1: 0.96
* **Drusen:** Precision: 0.91 | Recall: 0.83 | F1: 0.87


*├── data/
*│   ├── train/           # Eğitim veri seti (76,515 görüntü)
*│   │   ├── CNV/
*│   │   ├── DME/
*│   │   ├── DRUSEN/
*│   │   └── NORMAL/
*│   ├── val/             # Doğrulama veri seti (21,861 görüntü)
*│   │   ├── CNV/
*│   │   ├── DME/
*│   │   ├── DRUSEN/
*│   │   └── NORMAL/
*│   └── test/            # Test veri seti (10,933 görüntü)
*│       ├── CNV/
*│       ├── DME/
*│       ├── DRUSEN/
*│       └── NORMAL/
*├── metrics/
*│   └── f1score.py       # Özel F1-Skoru metrik sınıfı
*├── .gitattributes       # Git LFS veya özellik yapılandırmaları
*├── LICENSE              # Proje lisans dosyası
*├── Model_Prediction.ipynb # Tahmin ve görselleştirme işlemleri
*├── Trained_Eye_Disease_model.h5 # Kayıtlı model (Legacy H5 formatı)
*├── Trained_Eye_Disease_model.keras # Kayıtlı model (Yeni Keras formatı)
*├── Training_history.pkl # Eğitim logları (Loss/Accuracy verileri)
*└── Training_Model.ipynb # Modelin eğitim süreci kaynak kodu


## 🛠️ Installation & Usage

### Prerequisites
Ensure you have Python 3.10+ and the following libraries installed:
```bash
pip install tensorflow matplotlib seaborn pandas scikit-learn
