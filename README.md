# AI-Based Predictive Maintenance System for Real-Time Fault Diagnosis of Rotating Machinery

## Overview

This project presents a low-cost AI-based predictive maintenance system for rotating machinery using vibration analysis, machine learning, and embedded sensing.

A custom test rig consisting of a 120 mm dual-ball-bearing DC fan, ESP32 microcontroller, and MPU6050 triaxial accelerometer was developed to collect vibration data under multiple fault conditions. The acquired signals are processed through a feature engineering pipeline and classified using machine learning models to identify machine faults in real time.

The system is capable of detecting:

* Normal Operation
* Mass Imbalance (Mild, Moderate, Severe)
* Mechanical Looseness (Mild, Moderate, Severe)
* Blade Damage
* Bearing Contamination
* Air Obstruction

---

## System Architecture

![Architecture](images/architecture_diagram.png)

The complete pipeline consists of:

1. Vibration acquisition using MPU6050
2. Data transmission through ESP32
3. Feature extraction from vibration windows
4. Machine learning classification
5. Real-time fault prediction

---

## Hardware Setup

### Components

| Component     | Description                        |
| ------------- | ---------------------------------- |
| ESP32         | Data acquisition and communication |
| MPU6050       | Triaxial vibration sensor          |
| 120 mm DC Fan | Rotating machine under test        |
| PC            | Feature extraction and inference   |

![Hardware Setup](images/hardware_setup.jpg)

---

## Dataset

The dataset consists of vibration recordings collected under ten operating conditions.

| Class | Description           |
| ----- | --------------------- |
| 0     | Normal                |
| 1     | Imbalance Mild        |
| 2     | Imbalance Moderate    |
| 3     | Imbalance Severe      |
| 5     | Blade Damage          |
| 7     | Looseness Mild        |
| 8     | Looseness Moderate    |
| 9     | Looseness Severe      |
| 11    | Bearing Contamination |
| 13    | Air Obstruction       |

Total dataset size:

* 7,443 vibration windows
* Window length: 1024 samples
* Sampling rate: approximately 595 Hz

---

## Feature Engineering

Twenty-seven vibration features were extracted:

### Time Domain Features

* RMS
* Peak
* Crest Factor
* Peak-to-Peak
* Kurtosis
* Skewness
* Shape Factor
* RMS Ratios
* Autocorrelation Features

### Frequency Domain Features

* 1× Harmonic Amplitude
* 2× Harmonic Amplitude
* 3× Harmonic Amplitude
* Harmonic Ratios
* Spectral Entropy

---
## Machine Learning Models

Three different machine learning and deep learning models were investigated to evaluate their suitability for rotating machinery fault diagnosis.

### Random Forest
Random Forest served as the baseline ensemble learning model due to its robustness, interpretability, and strong performance on vibration-based classification tasks. It provided excellent accuracy while requiring minimal feature preprocessing.

### Extra Trees (Final Deployment Model)
The Extra Trees (Extremely Randomized Trees) classifier achieved the best balance between classification accuracy, inference speed, and computational efficiency. Owing to its lightweight architecture and fast prediction time, it was selected as the final model for deployment in the real-time monitoring system.

### Deep Neural Network (DNN)
A fully connected Deep Neural Network was implemented as a comparative deep learning model. Although it achieved competitive classification performance, the higher computational complexity and memory requirements made it less suitable for real-time edge deployment compared to the Extra Trees classifier.

---

## Model Performance

The final Extra Trees classifier was trained using the 27-dimensional handcrafted feature set extracted from 7,443 labelled vibration windows representing ten operating conditions.

| Performance Metric | Value |
|--------------------|------:|
| **5-Fold Cross-Validation (F1-Macro)** | **99.96%** |
| **Test Accuracy** | **99.93%** |
| **Cross-Machine Accuracy** | **93.6%** |

These results demonstrate excellent classification performance while maintaining good generalization to an unseen fan, making the model suitable for practical predictive maintenance applications.

---

## Confusion Matrix

The confusion matrix shows excellent class-wise separability across all ten operating conditions with minimal misclassification between similar fault types.

<p align="center">
  <img src="images/confusion_matrix.png" width="650">
</p>

---

## Feature Importance

Feature importance analysis revealed that the classifier primarily relies on physically meaningful vibration characteristics rather than arbitrary statistical correlations.

The most influential features include:

- **Z/X RMS Ratio (`az_ax_rms_ratio`)**
- **X-axis RMS (`rms_ax`)**
- **Y-axis RMS (`rms_ay`)**
- **Y/X RMS Ratio (`ay_ax_rms_ratio`)**
- **Z-axis RMS (`rms_az`)**
- **Z-axis Autocorrelation (1 Revolution)**

These features effectively distinguish imbalance, looseness, blade damage, bearing contamination, and air obstruction conditions.

<p align="center">
  <img src="images/feature_importance.png" width="650">
</p>

---

## Real-Time Fault Detection

The trained Extra Trees classifier is integrated with a Python-based real-time inference framework connected to an ESP32 and MPU6050 triaxial accelerometer.

### Real-Time Pipeline

- Live vibration acquisition from the ESP32
- Sliding-window segmentation (1024 samples)
- Feature extraction using the same 27-feature pipeline as training
- Feature standardization using the trained scaler
- Real-time fault prediction using the Extra Trees classifier
- Majority-voting over consecutive windows for stable predictions
- Continuous monitoring and fault reporting

The system is capable of identifying all ten operating conditions in real time, making it suitable for low-cost edge-based predictive maintenance applications.

<p align="center">
  <img src="images/realtime_prediction.png" width="750">
</p>
## Real-Time Fault Detection

The trained model is integrated with a real-time Python inference framework.

Features:

* Live vibration acquisition
* Sliding-window analysis
* Majority-voting prediction
* Continuous fault monitoring

![Real-Time Prediction](images/realtime_prediction.png)

---

## Repository Structure

```text
firmware/
feature_engineering/
model_training/
realtime_inference/
models/
results/
images/
docs/
```

---

## Future Improvements

* True bearing fault dataset (BPFO/BPFI)
* Variable-speed operation
* TinyML deployment
* Multi-sensor fusion
* Cloud-based monitoring dashboard

---

## Author

Ankit Banerjee

B.Tech, Electronics and Communication Engineering

North-Eastern Hill University (NEHU)

Shillong, India

