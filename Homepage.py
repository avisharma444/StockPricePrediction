import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st
from pandas_datareader import data as pdr
from datetime import datetime
import warnings

yf.pdr_override()

sns.set(rc={'figure.figsize': (20, 15)})
warnings.filterwarnings('ignore')

# Data utility functions
def stock_data(stock_name, duration, company_name):  # duration in years
    end_time = datetime.now()
    start_time = datetime(datetime.now().year - duration, datetime.now().month, datetime.now().day)
    stock = yf.download(stock_name, start_time, end_time)
    stock["company_name"] = company_name
    return stock

def plot(df, company):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Adj Close'], color='blue', label='Adjusted Close')
    plt.title(f'Adjusted Close Prices for {company} Inc.')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Streamlit app
st.title("Stock Analysis App")

# Select stock
stocks = {"AAPL": "Apple Inc.", "SPY": "S&P 500 ETF", "INFY": "Infosys", "^NSEI": "Nifty 50"}
selected_stock = st.selectbox("Select the stock for prediction:", list(stocks.keys()))

# Download data
stock_name = selected_stock
company_name = stocks[selected_stock]
apple_stocks = stock_data(stock_name, 10, company_name)

# Plot adjusted close prices
st.subheader(f"Adjusted Close Prices for {company_name}")
plot(apple_stocks, company_name)

# Filter data for the last year
last_year_data = apple_stocks.loc[apple_stocks.index >= datetime.now() - pd.DateOffset(years=1)]

# Calculate daily returns
last_year_data['Daily Return'] = last_year_data['Close'].pct_change()

# Plot daily returns
st.subheader("Daily Returns for the Past Year")
plt.figure(figsize=(12, 5))
plt.plot(last_year_data['Daily Return'], linestyle='--', marker='o')
plt.title(f"Daily returns for {company_name} in the past year")
st.pyplot(plt)

# Function to calculate and plot moving averages
def calculate_and_plot_moving_averages(data, periods):
    for p in periods:
        column_name = f"{p} days"
        data[column_name] = data['Close'].rolling(window=p).mean()
    st.subheader("Moving Averages")
    plt.figure(figsize=(12, 10))
    data[['Close'] + [f'{p} days' for p in periods]].plot(subplots=False)
    st.pyplot(plt)

# Calculate and plot moving averages
periods = [10, 20]
calculate_and_plot_moving_averages(last_year_data, periods)

# Display head of the DataFrame
st.subheader("Head of the DataFrame")
st.write(apple_stocks.head(10))

# Plot candlestick chart with custom colors
st.subheader(f"{company_name} Stock Candlestick Chart")
fig = go.Figure(data=[go.Candlestick(x=apple_stocks.index,
                open=apple_stocks['Open'],
                high=apple_stocks['High'],
                low=apple_stocks['Low'],
                close=apple_stocks['Close'],
                increasing_line_color='green', decreasing_line_color='red')])
fig.update_layout(
    title=f'{company_name} Stock Candlestick Chart',
    yaxis_title=company_name,
)
st.plotly_chart(fig)

# Create a heatmap of the correlation between stock price features
st.subheader("Correlation Between Features")
corr = apple_stocks[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Between Features')
st.pyplot(plt)
