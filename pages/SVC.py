import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

yf.pdr_override()
st.title("Stock Prediction Application")
st.subheader("Strategy Returns using Support Vector Classifiers")

# User inputs
stocks = ("AAPL", "GOOG", "MSFT", "^NSEI")  # Add more stocks as needed
selected_stocks = st.selectbox("Select the stock for prediction: ", stocks)
# n_months = st.slider("Months you want prediction for: ", 1, 5)

def stock_data(ticker, years, name):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Name'] = name
    return data

# Load data
data_load_state = st.text("Loading Data...")
apple_stocks = stock_data(selected_stocks, 4, selected_stocks)
data_load_state.text("Data Loaded Successfully!")

def time_series_cross_validation(df, one_hot_encodings, p):
    row = int(df.shape[0] * p)
    val_row = int((df.shape[0] - row) * 0.5) + row

    x_train = df[:row]
    y_train = one_hot_encodings[:row]

    x_val = df[row:val_row]
    y_val = one_hot_encodings[row:val_row]

    x_test = df[val_row:]
    y_test = one_hot_encodings[val_row:]

    return x_train, y_train, x_val, y_val, x_test, y_test

def create_features(apple_stocks):
    apple_stocks["Daily-Max-Fluctuation"] = apple_stocks["High"] - apple_stocks["Low"]
    apple_stocks["Daily-Fluctuation"] = apple_stocks["Open"] - apple_stocks["Close"]
    apple_stocks['Day'] = apple_stocks.index.day
    return apple_stocks[["Daily-Fluctuation", "Daily-Max-Fluctuation", "Close"]], apple_stocks

def plot_daily_fluctuations(apple_stocks):
    sns.boxplot(data=apple_stocks, x="Day", y="Daily-Fluctuation")
    plt.title('Daily Fluctuations')
    plt.xlabel('Day of the Month')
    plt.ylabel('Daily Fluctuation')
    st.pyplot(plt.gcf())

def plot_returns(apple_stocks):
    apple_stocks['return'] = apple_stocks['Close'].pct_change(1)
    apple_stocks['shifted_predictions'] = apple_stocks['predictions'].shift(1)
    apple_stocks['strategy_returns'] = apple_stocks['shifted_predictions'] * apple_stocks['return']

    ret = [0]
    strat = [0]
    for i in range(1, len(apple_stocks)):
        ret.append(ret[-1] + apple_stocks['return'].iloc[i])
        strat.append(strat[-1] + apple_stocks['strategy_returns'].iloc[i])
    apple_stocks['ret'] = ret
    apple_stocks['strat'] = strat

    # Ensure the index is a datetime index with timezone information
    if apple_stocks.index.tz is None:
        apple_stocks.index = apple_stocks.index.tz_localize('UTC')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(apple_stocks.index, apple_stocks['ret'], color='red', label='Predicted Strategy Returns')
    ax.plot(apple_stocks.index, apple_stocks['strat'], color='blue', label='Actual Stock Returns')
    ax.set_title('Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend()

    st.pyplot(fig)

def SVC_Classifier(apple_stocks, x_train, y_train, x_val, y_val, x_test, y_test, X):
    model = SVC()
    model.fit(x_train[["Daily-Fluctuation", "Daily-Max-Fluctuation"]], y_train)

    train_err = model.score(x_train[["Daily-Fluctuation", "Daily-Max-Fluctuation"]], y_train)
    test_err = model.score(x_test[["Daily-Fluctuation", "Daily-Max-Fluctuation"]], y_test)
    val_err = model.score(x_val[["Daily-Fluctuation", "Daily-Max-Fluctuation"]], y_val)

    apple_stocks['predictions'] = model.predict(X[["Daily-Fluctuation", "Daily-Max-Fluctuation"]])

    st.write("The model's score for the Test Set came out to be : ", test_err)
    st.write("The model's score for the Validation Set came out to be : ", val_err)
    plot_returns(apple_stocks)

# Streamlit app
st.sidebar.header('Settings')
years = st.sidebar.slider('Years of data', 1, 10, 4)
test_percentages = st.sidebar.multiselect('Select train-test splits', [0.2, 0.4, 0.6, 0.8], [0.2])

# Initial load of data
temp, apple_stocks = create_features(apple_stocks)
buy_or_sell = np.where(apple_stocks.Close.shift(-1) > apple_stocks.Close, 1, 0)

for p in test_percentages:
    st.write(f"Train - Test Split : {(p)*100} - {(1-p)*100}")
    x_train, y_train, x_val, y_val, x_test, y_test = time_series_cross_validation(apple_stocks, buy_or_sell, p)
    SVC_Classifier(apple_stocks, x_train, y_train, x_val, y_val, x_test, y_test, temp)
