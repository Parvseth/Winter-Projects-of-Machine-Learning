# NAME : PARV SETH
# ROLL NUMBER : 23B0652
## Description of Work

- **Week 2**  
  **Language**: Python  
  Built a gradient descent model to predict Coronary Heart Disease.
  # ❤️ Heart Disease Prediction using Enhanced Logistic Regression

A comprehensive implementation of logistic regression from scratch to predict heart disease risk using the **Framingham Heart Study** dataset, blending rigorous machine learning techniques with clinical insights and professional evaluation metrics.

---

## 🧠 Project Overview

This project applies an **improved logistic regression model** to predict 10-year coronary heart disease (CHD) risk from real patient health indicators. The model is built from scratch in Python, featuring:
- Robust optimization and convergence control  
- Feature scaling and regularization  
- Cross-validation and test evaluation  
- Clinical interpretability and data visualization  

> ⚡ **Highlight**: 85.1% cross-validation accuracy, AUC up to 0.88, professional ML pipeline, strong real-world applicability.

---

## 📊 Dataset: Framingham Heart Study

| Detail                | Value                        |
|----------------------|------------------------------|
| Source               | Framingham Heart Study        |
| Records              | 4,238                         |
| Target               | 10-year CHD risk (binary)     |
| Features             | 15 health metrics             |
| Challenges           | Missing values, class imbalance |

### 🧬 Feature Types
- **Demographics**: Age, gender  
- **Behavioral**: Smoking status, cigarettes/day  
- **Medical History**: Stroke, hypertension, diabetes  
- **Vitals/Labs**: BP, heart rate, cholesterol, glucose  
- **Target**: `TenYearCHD` (0 = No, 1 = Yes)

---

## 🔍 Enhanced Model Architecture

### 📌 Logistic Regression (From Scratch)
```math
P(y=1 | x) = \sigma(w_0 + w_1x_1 + ... + w_nx_n)
where \sigma(z) = \frac{1}{1 + e^{-z}}
```
Mean Accuracy: 0.8547 ± 0.0124
```
### 🔍 Feature Importance

```
| Rank | Feature           | Importance |
|------|-------------------|------------|
| 1    | Age               | High       |
| 2    | CigsPerDay        | High       |
| 3    | SysBP             | High       |
| 4    | Male              | Medium     |
| 5    | TotChol           | Medium     |
| 6    | PrevalentHyp      | Medium     |
| 7    | PrevalentStroke   | Low        |
| 8    | Glucose           | Low        |
| 9    | CurrentSmoker     | Low        |
| 10   | Diabetes          | Low        |


Install dependencies :
```
pip install numpy pandas matplotlib seaborn scikit-learn
```
Configuration : 
```
model_params = {
    'learning_rate': 0.01,
    'max_iterations': 2000,
    'regularization': 'l2',
    'lambda_reg': 0.01,
    'tolerance': 1e-6
}
```
## 🏥 Clinical Significance

- **Preventive Screening**: Early detection of high-risk patients  
- **Cost-Efficiency**: Targeted healthcare intervention  
- **Transparent AI**: Interpretability for doctors  
- **Alignment with Literature**: Feature influence matches medical research  

---

## 📈 Advanced Features

### 🧠 Data Science Excellence
- Stratified Cross-Validation  
- Missing Value Imputation  
- Feature Standardization  
- L1/L2 Regularization  

### 📊 Professional Evaluation
- ROC-AUC Curves  
- Confusion Matrices  
- Feature Importance Visuals  
- F1, Recall, Precision metrics  

### ⚙️ Production-Ready Code
- Modular design  
- Logging and configuration  
- Ready for REST API wrapping  
- Reproducible training  

---

## 🔮 Future Enhancements

- 🔍 Hyperparameter Tuning (Grid/Bayesian search)  
- 🌐 REST API & Web UI for Deployment  
- 🧠 Ensemble Methods (XGBoost, Random Forest)  
- 📉 Precision-Recall Curves for better imbalance handling  
- 📦 Model Persistence & Loading  



- **Week 3**  
  **Language**: Python (PyTorch)  
  Built a model to categorize handwritten digits using the MNIST dataset.

  # MNIST Handwritten Digit Classification

An advanced PyTorch implementation of a fully connected neural network for classifying handwritten digits from the MNIST dataset with state-of-the-art techniques.

## Project Overview

This project implements an improved neural network architecture to classify handwritten digits (0-9) from the MNIST dataset. The enhanced model achieves **98% accuracy** on the test set, representing a significant improvement over basic implementations through advanced training techniques and architectural improvements.

## Model Architecture

The enhanced neural network features a modern deep learning architecture with regularization:

- **Input Layer**: 784 neurons (28×28 flattened pixel values)
- **Hidden Layer 1**: 128 neurons with BatchNorm → ReLU → Dropout(0.3)
- **Hidden Layer 2**: 64 neurons with BatchNorm → ReLU → Dropout(0.3)
- **Hidden Layer 3**: 32 neurons with BatchNorm → ReLU → Dropout(0.3)
- **Output Layer**: 10 neurons (for 10 digit classes)

```
Input (784) → FC(128) → BatchNorm → ReLU → Dropout → 
FC(64) → BatchNorm → ReLU → Dropout → 
FC(32) → BatchNorm → ReLU → Dropout → 
FC(10) → Softmax
```

**Key Architectural Improvements:**
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Dropout Regularization**: Prevents overfitting with 30% dropout rate
- **Progressive Layer Reduction**: Efficient feature extraction (128→64→32)
- **Kaiming Weight Initialization**: Optimized for ReLU activations

## Dataset

- **Source**: MNIST handwritten digit dataset stored in `MNIST_data.pkl`
- **Original Size**: 60,000 training samples
- **Format**: 28×28 grayscale images flattened to 784-dimensional vectors
- **Classes**: 10 classes (digits 0-9)
- **Data Split**: 80% training, 10% validation, 10% testing
- **Preprocessing**: Normalized to [0,1] range
- **Data Augmentation**: Noise injection increases training set to 96,000 samples

## Advanced Training Configuration

- **Optimizer**: Adam with learning rate 0.001 and weight decay 1e-5
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 64 (improved from 10 for better gradient estimates)
- **Max Epochs**: 20 with early stopping
- **Regularization**: 
  - Dropout (30% rate)
  - Batch normalization
  - L2 weight decay
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping**: Patience=7 epochs to prevent overfitting
- **Data Augmentation**: Gaussian noise injection (σ=0.05)

## Key Features

### Advanced Architecture
- **Modern Deep Learning**: Batch normalization + dropout for robust training
- **Dynamic Architecture**: Configurable layer sizes through parameters
- **GPU Support**: Automatic CUDA detection and utilization
- **Professional Weight Initialization**: Kaiming initialization for optimal gradient flow

### Smart Training
- **Validation Monitoring**: Separate validation set for hyperparameter tuning
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Comprehensive Logging**: Detailed training progress tracking

### Data Enhancement
- **Normalization**: Pixel values scaled to [0,1] for better convergence
- **Data Augmentation**: Noise injection doubles effective training data
- **Efficient Loading**: PyTorch DataLoader with proper batching and shuffling
- **Reproducible Results**: Fixed random seeds for consistent results

## Results

### Performance Metrics
```
Final Test Accuracy: 98.0%
Model Parameters: 113,322 trainable parameters

Classification Report:
              precision    recall  f1-score   support
           0       0.98      0.98      0.98       599
           1       0.99      0.98      0.98       588
           2       0.99      0.98      0.98       575
           3       0.98      0.98      0.98       565
           4       0.98      0.99      0.98       558
           5       0.98      0.98      0.98       523
           6       0.99      0.99      0.99       593
           7       0.98      0.98      0.98       653
           8       0.97      0.98      0.98       590
           9       0.98      0.98      0.98       606

    accuracy                           0.98      6000
   macro avg       0.98      0.98      0.98      6000
weighted avg       0.98      0.98      0.98      6000
```

### Training Progress
The model demonstrates excellent convergence with:
- **Smooth Loss Reduction**: Training loss drops from ~0.6 to ~0.1
- **Stable Validation**: No signs of overfitting, validation accuracy reaches ~97%
- **Consistent Improvement**: Steady accuracy gains across all epochs
- **Balanced Performance**: All digit classes achieve >97% precision and recall

### Confusion Matrix Analysis
The confusion matrix reveals:
- **Strong Diagonal**: High true positive rates for all classes
- **Minimal Confusion**: Very few misclassifications between similar digits
- **Balanced Errors**: No particular class dominates the error patterns

## Code Structure

### Enhanced Neural Network Class
```python
class ImprovedNet(nn.Module):
    def __init__(self, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        # Dynamic architecture with batch norm and dropout
        
    def forward(self, x):
        # Forward pass through sequential layers
```

### Professional Training Pipeline
1. **Data Loading & Preprocessing**: Normalization and augmentation
2. **Smart Data Splitting**: Train/validation/test with proper stratification  
3. **Advanced Training**: Early stopping, LR scheduling, validation monitoring
4. **Comprehensive Evaluation**: Multiple metrics, confusion matrix, visualization

### Monitoring & Visualization
- **Training Curves**: Loss and accuracy plots for training/validation
- **Confusion Matrix**: Detailed error analysis per class
- **Classification Report**: Precision, recall, F1-score for each digit
- **Model Persistence**: Save/load trained models

## Dependencies

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

**Core Libraries:**
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization and plotting
- **Scikit-learn**: Metrics and evaluation tools

## Usage

1. **Setup**: Ensure all dependencies are installed
2. **Data Preparation**: Place `MNIST_data.pkl` in the specified directory
3. **Configuration**: Modify the CONFIG dictionary for custom hyperparameters
4. **Training**: Run the script to train with automatic validation monitoring
5. **Evaluation**: View comprehensive metrics, plots, and confusion matrix
6. **Model Saving**: Trained model automatically saved as `improved_mnist_model.pth`

```python
# Key configuration options
CONFIG = {
    'batch_size': 64,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'hidden_sizes': [128, 64, 32],
    'dropout_rate': 0.3,
    'augment_data': True
}
```

## Performance Improvements Achieved

| Metric | Original Model | Improved Model | Improvement |
|--------|---------------|----------------|-------------|
| **Test Accuracy** | 94.39% | 98.00% | +3.61% |
| **Architecture** | Simple 4-layer | Advanced with BatchNorm/Dropout | Modern techniques |
| **Training Strategy** | Basic 3 epochs | Smart early stopping | Prevents overfitting |
| **Data Usage** | Raw pixels | Normalized + augmented | Better generalization |
| **Validation** | None | Dedicated val set | Proper hyperparameter tuning |
| **Monitoring** | Basic loss | Comprehensive metrics | Professional evaluation |

## Advanced Features Implemented

✅ **Batch Normalization** - Stabilizes training and improves convergence  
✅ **Dropout Regularization** - Prevents overfitting  
✅ **Data Augmentation** - Improves generalization with noise injection  
✅ **Early Stopping** - Automatic training termination  
✅ **Learning Rate Scheduling** - Adaptive learning rate reduction  
✅ **GPU Support** - Automatic CUDA utilization  
✅ **Comprehensive Evaluation** - Classification report, confusion matrix  
✅ **Professional Visualization** - Training curves and error analysis  
✅ **Model Persistence** - Save and reload trained models

## Performance Analysis

The **98% accuracy** represents excellent performance for a fully connected network on MNIST:

### Training Characteristics
- **Rapid Convergence**: Loss drops dramatically in first few epochs
- **Stable Learning**: Smooth training and validation curves with no overfitting
- **Balanced Performance**: All digits achieve >97% precision/recall
- **Efficient Training**: Early stopping prevented unnecessary computation

### Model Insights
- **Optimal Architecture**: Progressive layer reduction (128→64→32) efficiently captures features
- **Effective Regularization**: 30% dropout prevents overfitting without hampering performance  
- **Strong Generalization**: High validation accuracy indicates good generalization
- **Minimal Confusion**: Clean confusion matrix with very few off-diagonal errors

### Technical Excellence
The implementation demonstrates production-ready deep learning practices with professional monitoring, comprehensive evaluation, and robust training procedures.

---

*This enhanced implementation showcases modern deep learning techniques applied to the classic MNIST problem, achieving near state-of-the-art performance for fully connected networks while maintaining code clarity and professional standards.*


- **Week 4**  
  **Language**: Python (PyTorch, Torchvision)  
  Used a CNN model for binary image classification. The model was trained on the Cats vs. Dogs dataset.

- **CIFAR-10**  
  **Language**: Python (PyTorch, Torchvision)  
  Used a CNN model for multi-class classification to categorize images into one of the 10 categories in the CIFAR-10 dataset.

  # 📦 CIFAR-10 Image Classifier using PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images in 10 different classes, split into 50,000 training images and 10,000 test images.

---

## 🗂️ Dataset Description

The data is stored in the folder:  
`D:\wids '23\Winter-Projects-of-Machine-Learning\cifar-10-batches-py`

### Files:
- `data_batch_1` to `data_batch_5`: Training data
- `test_batch`: Test data
- `batches.meta`: Contains class label names
- `readme`: Dataset description

Each batch is a Python `pickle` file containing a dictionary with:
- `'data'`: image data, shape `(10000, 3072)`  
- `'labels'`: list of image labels  
- `'filenames'`: image file names

---

## 🧠 Model Architecture

A simple CNN is implemented with:
- 2 Convolutional layers:
  - `Conv2d(3, 6, 5)` → ReLU → MaxPool(2x2)
  - `Conv2d(6, 16, 5)` → ReLU → MaxPool(2x2)
- 3 Fully connected layers:
  - `Linear(400, 120)` → ReLU
  - `Linear(120, 84)` → ReLU
  - `Linear(84, 10)` (for 10 classes)

---

## 🧪 Training Setup

- **Epochs:** 5  
- **Batch Size:** 4  
- **Optimizer:** SGD  
- **Loss Function:** CrossEntropyLoss  
- **Learning Rate:** 0.001  
- **Device:** GPU if available, else CPU  

---

## 🧰 Dependencies

Install using pip:

```bash
pip install torch torchvision matplotlib

```
'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
```
Accuracy of the network: 52.3 %
Accuracy of plane: 60.7 %
Accuracy of car: 64.2 %
Accuracy of bird: 41.2 %
Accuracy of cat: 30.6 %
Accuracy of deer: 48.4 %
Accuracy of dog: 37.5 %
Accuracy of frog: 44.7 %
Accuracy of horse: 59.8 %
Accuracy of ship: 77.8 %
Accuracy of truck: 47.6 %



