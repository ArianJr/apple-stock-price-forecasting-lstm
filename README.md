# 🍎 Apple Stock Price Forecasting with LSTM

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![Colab](https://img.shields.io/badge/Open%20in-Colab-orange.svg)](https://colab.research.google.com/github/ArianJr/apple-stock-price-forecasting-lstm/blob/main/notebooks/Apple_Stock_Price_Forecasting.ipynb)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

---

This project uses a **Long Short-Term Memory (LSTM)** neural network to predict Apple Inc. (AAPL) stock closing prices from historical data.  
It demonstrates how **deep learning can be applied to time-series financial forecasting** in a reproducible, end-to-end pipeline.

A deep-learning project that analyzes and forecasts Apple Inc. (AAPL) stock prices using Long Short-Term Memory (LSTM) neural networks. This notebook demonstrates a complete end-to-end workflow including data preprocessing, exploratory analysis, model construction, training, evaluation, and visualization of forecasting results.

This project is ideal for:
- Machine learning and deep learning portfolios
- Financial time-series analysis demonstrations
- Model forecasting showcases
- GitHub repositories that highlight practical ML applications

---

## 🚀 Project Overview

The goal of this project is to build an LSTM-based model capable of predicting Apple’s opening stock prices using historical data. LSTMs are particularly powerful for time-series tasks due to their ability to learn long-range temporal dependencies.

The notebook includes:

- Exploratory Data Analysis (EDA)
- Data scaling & sequence generation
- LSTM model architecture and training
- Prediction and inverse scaling
- RMSE evaluation
- Visual comparison between actual and predicted values
- Loss curve visualization

---

## ⚡ Key Features

- End-to-end deep learning pipeline using TensorFlow/Keras
- Historical Apple stock data preprocessing
- Sliding window sequence creation (60-day input window)
- LSTM stacked architecture for sequential learning
- RMSE evaluation metric
- High-quality visualizations of training progress and predictions
- Reproducible results using fixed random seeds

---

## 📐 Architecture Overview
Historical Stock Data → Data Preprocessing → Sequence Generation → LSTM Model → Forecasted Price → Evaluation & Visualization

---

## 🧠 Model Architecture

The forecasting model is built using a Long Short-Term Memory (LSTM) neural network. Here's a breakdown of the architecture used in the notebook:

A multi-layer LSTM model including:
- LSTM layers
- Dropout
- Dense output layer
- Mean Squared Error (MSE) loss
- Adam optimizer

Trained on sequences of 60 time steps.

- Input Layer
  - The model takes sequences of past 60 days of normalized closing price data (window length), forming the input shape (sequence_length, 1).
- LSTM Layers
  - First LSTM layer
    - Units: 300 
    - return_sequences=True, so that it outputs a full sequence to the next LSTM.
    - Activation: typically tanh (default) for LSTM.
    - Dropout: e.g., Dropout(0.2) to regularize and reduce overfitting.
  - Second LSTM layer
    - Units: 100 
    - return_sequences=True
    - Activation: tanh
    - Dropout: 0.2
  - Third LSTM layer
    - Units: 100 
    - return_sequences=True
    - Activation: tanh
    - Dropout: 0.2
  - Forth LSTM
    - Units: 100 
    - return_sequences=False
    - Activation: tanh
    - Dropout: 0.2
- Dense Output Layer
  - After the LSTM layers, there's a fully connected (Dense) layer with 1 neuron, producing the forecast for the next day's closing price.
  - Activation: linear, as this is a regression task.
- Compilation
  - Loss Function: Mean Squared Error (MSE) — well-suited for regression and time-series prediction.
  - Optimizer: Adam — commonly used for its good convergence behavior.


### Model Summary

| Layer (type)       | Output Shape     | Param #  |
|--------------------|------------------|----------|
| LSTM               | (None, 60, 300)  | 362,400  |
| Dropout            | (None, 60, 300)  | 0        |
| LSTM               | (None, 60, 100)  | 160,400  |
| Dropout            | (None, 60, 100)  | 0        |
| LSTM               | (None, 60, 100)  | 80,400   |
| Dropout            | (None, 60, 100)  | 0        |
| LSTM               | (None, 100)      | 80,400   |
| Dropout            | (None, 100)      | 0        |
| Dense              | (None, 1)        | 101      |

---

## 📊 Results

### 📉 Apple Stock Price: Actual vs Predicted

![Actual vs Predicted Apple Stock Price](images/actual_vs_predicted_apple_stock_price)


### 📉 Loss Over Epochs
![Loss Over Epochs](images/loss_ovre_epochs)

---

## 🛠 Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn

---

## 📐 Evaluation Metric

- RMSE (Root Mean Squared Error) is used to quantify prediction accuracy.
Lower RMSE indicates better performance.

| Metric | Value |
| ------ | ----- |
| RMSE   | ~4.7  |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+  
- pip or conda  
- Required libraries: `tensorflow`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`

### Installation
```bash
git clone https://github.com/ArianJr/apple-stock-price-forecasting-lstm.git
cd apple-stock-price-forecasting-lstm
pip install -r requirements.txt
```

### Running the Notebook
1. Open: `apple_stock_price_forecasting_lstm.ipynb`
2. Run cells step-by-step
```bash
# Load libraries & data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
```
3. Preprocess the data → Build LSTM → Train & Evaluate → Visualize predictions

---

## 📦 Dataset Source

This project uses the [Apple Stock (2013–2018) dataset](https://www.kaggle.com/datasets/soheiltehranipour/apple-stock-20132018/data) by Soheil Tehranipour on Kaggle.

**Fields Used:**  
- `Date`: Trading date  
- `Open`: Opening price of Apple stock

**Note:** This dataset contains only two columns and may require augmentation for multi-feature modeling.

---

## 📚 Future Improvements

- Incorporating OHLCV input features
- Adding dropout + recurrent dropout for improved generalization
- Using GRU or Transformer-based architectures
- Hyperparameter tuning (learning rate, batch size, number of units)
- Live model deployment using FastAPI or Streamlit

---

## 🔧 Customization & Extensions

- Switch Stocks: Change ticker symbol (MSFT, GOOGL, etc.)
- Forecast Horizon: Adjust sequence length or future window
- Model Architectures: Try GRU, CNN, or Transformers
- Hyperparameter Tuning: Modify layers, hidden units, dropout, batch size, epochs

---

## 🙏 Acknowledgements

- Thanks to **soheiltehranipour** for the Apple stock dataset (2013–2018) on Kaggle: [Apple Stock (2013–2018) dataset](https://www.kaggle.com/datasets/soheiltehranipour/apple-stock-20132018/data)
- Inspired by numerous open‑source time-series forecasting repositories and tutorials — your work paved the way!  
- Built with help from the TensorFlow / Keras community — especially for LSTM guidance and best practices.  
- Thanks to all readers and contributors who test, critique, or improve this project.

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Arian Jr**  
📧 [Contact Me](arianjafar59@gmail.com) • 🌐 [GitHub Profile](https://github.com/ArianJr)

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/ArianJr" target="_blank">ArianJr</a>
</p>

<p align="center">
  <sub>⭐ If you found this project useful, please consider giving it a star! It helps others discover it and supports my work.</sub>
</p>

---

<p align="center">
  <img src="https://img.shields.io/github/stars/ArianJr/power-output-prediction-ann?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/ArianJr/power-output-prediction-ann?style=social" alt="GitHub forks">
</p>
