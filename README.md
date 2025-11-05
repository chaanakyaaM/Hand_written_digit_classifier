# Handwritten Digit Classifier 🧠✍️

A custom-trained Convolutional Neural Network (CNN) model recognizes handwritten digits (0–9) drawn on a simple Tkinter canvas interface.

## 📘 Project Overview
- This project allows users to draw digits on a canvas and predicts the number using a CNN model trained on MNIST dataset.  
- The training process is done in **`cnn.ipynb`**, and the GUI-based digit recognition is implemented in **`main.py`**.
- The **`CNN_200k_param_model.keras`** is the trained model used in `main.py`.
- Tkinter-based application that loads the model, takes user input from the canvas, and displays predictions in real time.  

## 🧠 Model Details  

### 🏗️ Architecture Overview  
The model is a **Convolutional Neural Network (CNN)** built using **TensorFlow/Keras** to classify handwritten digits (0–9) from the **MNIST dataset**.  
It takes a **28×28 grayscale image** as input and outputs probabilities for each digit class.

| Layer Type | Parameters | Activation | Output Shape | Description |
|-------------|-------------|-------------|---------------|--------------|
| **Conv2D** | 32 filters, (3×3) kernel | ReLU | (26, 26, 32) | Detects local spatial features in input images |
| **MaxPooling2D** | (2×2) | — | (13, 13, 32) | Reduces feature map dimensions to avoid overfitting |
| **Conv2D** | 64 filters, (3×3) kernel | ReLU | (11, 11, 64) | Extracts deeper visual patterns |
| **MaxPooling2D** | (2×2) | — | (5, 5, 64) | Further compresses spatial data |
| **Flatten** | — | — | (1600) | Converts 2D features into a flat vector |
| **Dense** | 128 units | ReLU | (128) | Fully connected layer for classification learning |
| **Dense (Output)** | 10 units | Softmax | (10) | Outputs probability distribution across 10 digits |

---

### ⚙️ Training Configuration  

| Parameter | Value |
|------------|--------|
| **Dataset** | MNIST (60,000 training / 10,000 testing images) |
| **Optimizer** | Adam |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Metrics** | Accuracy |
| **Epochs** | 10 |
| **Batch Size** | Default (32) |
| **Verbose** | 1 (Progress bar visible during training) |
| **Total Trainable Parameters** | 225,034 |

---

### 📈 Model Performance  

| Metric | Value |
|---------|--------|
| **Test Accuracy** | **98.50%** |
| **Test Loss** | **0.0588** |
| **Inference Speed** | < 0.1 sec per image (CPU) |

> The model trained for **10 epochs** using the **Adam optimizer**, achieving **98.5% accuracy** with a **loss of 0.0588** 
