# 📈 Apple Stock Price Forecasting Using LSTM

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

🧠 Key Features

- End-to-end deep learning pipeline using TensorFlow/Keras
- Historical Apple stock data preprocessing
- Sliding window sequence creation (60-day input window)
- LSTM stacked architecture for sequential learning
- RMSE evaluation metric
- High-quality visualizations of training progress and predictions
- Reproducible results using fixed random seeds

---

## 📊 Results

### 📉 Apple Stock Price: Actual vs Predicted

![Actual vs Predicted Apple Stock Price](INSERT_IMAGE_PATH_HERE)


### 📉 Loss Over Epochs
![Loss Over Epochs](INSERT_IMAGE_PATH_HERE)

---

## 🛠 Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn

---

## 🧪 Model Architecture

A multi-layer LSTM model including:
- LSTM layers
- Dropout
- Dense output layer
- Mean Squared Error (MSE) loss
- Adam optimizer

Trained on sequences of 60 time steps.

---

## 📐 Evaluation Metric

- RMSE (Root Mean Squared Error) is used to quantify prediction accuracy.
Lower RMSE indicates better performance.

---

## 📘 Usage

### 1. Clone the repository:
```bash
git clone https://github.com/ArianJr/apple-stock-price-forecasting-lstm.git
```
### 2. Install dependencies:
```bash
pip install -r requirements.txt
```
### 3. Run the notebook:
```bash
jupyter notebook apple_stock_price_forecasting_lstm.ipynb
```

---

## 📚 Future Improvements

- Incorporating OHLCV input features
- Adding dropout + recurrent dropout for improved generalization
- Using GRU or Transformer-based architectures
- Hyperparameter tuning (learning rate, batch size, number of units)
- Live model deployment using FastAPI or Streamlit

---

