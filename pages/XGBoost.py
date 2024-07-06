import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pandas_datareader import data as pdr
from datetime import datetime
import warnings
import xgboost as xgb
from sklearn import metrics
from resources.Evaluation_metrics import *

# Initialize settings
sns.set(rc={'figure.figsize': (20, 15)})
warnings.filterwarnings('ignore')
yf.pdr_override()

st.title("Stock Prediction Application")
st.subheader("XGBoost Model")

# User inputs
stocks = ("AAPL", "GOOG", "MSFT", "^NSEI")
selected_stocks = st.selectbox("Select the stock for prediction: ", stocks)
n_months = st.slider("Months you want prediction for: ", 1, 5)

# Function definitions
def stock_data(stock_name, duration, company_name):  # duration in years
    end_time = datetime.now()
    start_time = datetime(datetime.now().year - duration, datetime.now().month, datetime.now().day)  # using data of exactly duration year before
    stock = yf.download(stock_name, start_time, end_time)
    stock["company_name"] = company_name
    return stock

def feature_engineering(df, n=14):
    close = df['Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df['RSI'] = rsi.fillna(0)
    df['SMA_5'] = df['Close'].rolling(5).mean().shift()

def preprocess_data(df):
    l = len(df)
    split = 0.8
    train = df[:int(split * l)]
    test = df[int(split * l):]
    return train, test

def XBGoostRegression(train, test):
    train_features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_5']
    predictor_variables = ['Close']
    
    x_train = train[train_features]
    y_train = train[predictor_variables]
    
    x_test = test[train_features]
    y_test = test[predictor_variables]
    
    XGB = xgb.XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=3000, early_stopping_rounds=50, max_depth=6,
                           learning_rate=0.01, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1)
    XGB.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)
    return XGB, x_test

def evaluate_xgboost(XGB, test, df, x_test):
    predictions = XGB.predict(x_test)
    test['predictions'] = predictions
    return test, np.sqrt(metrics.mean_squared_error(test['Close'], test['predictions']))

# Loading data
data_load_state = st.text("Loading Data................")
stocks_data = stock_data(selected_stocks, 4, selected_stocks)
data_load_state.text("Loaded Data Successfully!")

# Display latest stock price data
st.subheader("Latest Stock Price Data")
st.write(stocks_data.tail())

# Feature engineering and data preparation
feature_engineering(stocks_data)
train, test = preprocess_data(stocks_data)

# Train the XGBoost model
XGB, x_test = XBGoostRegression(train, test)

# Evaluate the model
test_with_predictions, score = evaluate_xgboost(XGB, test, stocks_data, x_test)

# Plot predictions
st.write("Predictions using XGBoost Model")
fig, ax = plt.subplots(figsize=(15, 5))
stocks_data['Close'].plot(ax=ax, label='Actual Prices')
test_with_predictions['predictions'].plot(ax=ax, style='.', label='XGBoost Predictions')
plt.legend()
ax.set_title('Actual Past Data vs Predictions')
st.pyplot(fig)

# Evaluate metrics
y_true = test_with_predictions['Close']
y_pred = test_with_predictions['predictions']
styled_df = evaluate_metrics(y_true, y_pred)
st.write("Model Score for the last 30 day data on different evaluation metrics")
st.dataframe(styled_df)

st.write("The Mean Squared Error using XGBoost model was: ", score)
