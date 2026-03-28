
Overview
This project focuses on Automatic Modulation Classification (AMC) of wireless signals using machine learning. It uses raw in-phase and quadrature (I/Q) signal samples from the RadioML 2016.10a dataset and classifies them into different digital modulation schemes such as BPSK, QPSK, QAM16, QAM64, 8PSK, PAM4, CPFSK, GFSK, AM-DSB, AM-SSB, and WBFM.
The project builds a complete machine learning pipeline for wireless signal recognition, including data loading, preprocessing, feature extraction, model training, evaluation, signal visualization, dimensionality reduction using PCA, and feature importance analysis.

Problem Statement:
Modern wireless communication systems use multiple modulation schemes to transmit information efficiently. In practical communication environments, receivers often need to automatically identify the modulation type of a received signal, especially in systems such as cognitive radio, spectrum monitoring, and software-defined radio.
Traditional modulation classification methods rely heavily on handcrafted signal processing features and expert-designed rules. These methods often struggle in noisy and dynamic environments. This project solves that problem by using supervised machine learning models to classify modulation schemes directly from raw I/Q signal samples.

How It Works:
The project follows these steps:

1. Dataset Loading
   - Loads the RadioML 2016.10a dataset stored as a Python pickle file.
   - Each key in the dataset is a `(modulation type, SNR)` pair.
   - Each signal sample has shape `(2, 128)` representing I and Q components.

2. Data Understanding
   - Extracts all unique modulation types and SNR levels.
   - The dataset contains:
     - 11 modulation classes
     - 20 SNR levels

3. Preprocessing
   - Each signal sample is flattened from `2 × 128` into a `256-dimensional` feature vector.
   - A subset of **200 samples per (modulation, SNR) pair** is used.
   - Final dataset size:
     - 44,000 total samples
     - `X shape = (44000, 256)`
     - `y shape = (44000,)`

4. Label Encoding
   - Converts modulation labels into numeric values using `LabelEncoder`.

5. Train-Test Split
   - Splits the dataset into:
     - 80% training
     - 20% testing
   - Uses stratified sampling to preserve class balance.

6. Feature Scaling
   - Standardizes input features using `StandardScaler`.

7. Model Training
   - Trains multiple supervised learning models:
     - K-Nearest Neighbors (KNN)
     - Linear SVM
     - Logistic Regression
     - Decision Tree
     - Random Forest

8. Evaluation
   - Evaluates models using:
     - Accuracy
     - Classification Report
     - Confusion Matrix

9. Signal Visualization
   - Plots time-domain I/Q components.
   - Generates constellation diagrams for sample signals.

10. PCA-Based Dimensionality Reduction
    - Reduces feature size from `256` to `50` dimensions.
    - Tests the impact of PCA on Random Forest performance.

11. Feature Importance Analysis
    - Uses Random Forest feature importance to study the contribution of I and Q channel time steps.

Results

Model Accuracy
- KNN:27.72%
- Linear SVM:15.42%
- Logistic Regression:15.26%
- Decision Tree: 21.23%
- Random Forest: 36.51%

# Best Model
- Random Forest achieved the highest classification accuracy and performed best among the implemented models.

PCA Result:
- Random Forest without PCA: 36.07%
- Random Forest with PCA:33.52%

This shows that PCA reduced dimensionality with only a moderate drop in accuracy.

 Tech Used
- Python
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle
- Jupyter Notebook

Expected dataset file locations in the notebook:
- `/content/drive/MyDrive/colab_data/RML2016.10a_dict.pkl`
- `../data/RML2016.10a_dict.pkl`
- `data/RML2016.10a_dict.pkl`

How to Run

1. Install dependencies
```bash
pip install numpy scikit-learn matplotlib seaborn notebook
