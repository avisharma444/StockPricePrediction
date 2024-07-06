# %%
import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
yf.pdr_override()
from datetime import datetime

# %%
def stock_data(stock_name, duration, company_name):#duration in years
   
    end_time = datetime.now()
    start_time = datetime(datetime.now().year - duration, datetime.now().month, datetime.now().day)  #using data of exactly duration year before
    stock=yf.download(stock_name,start_time,end_time)
    stock["comany_name"]=company_name
    return stock

# %%
def plot(df, company):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Adj Close'], color='blue', label='Adjusted Close')
    plt.title(f'Adjusted Close Prices for {company} Inc.')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()  
    plt.grid(True)
    plt.show()

# # %%
# apple_stocks=stock_data('AAPL', 4,'Apple')
# plot(apple_stocks,'Apple')

