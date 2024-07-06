
import streamlit as st

st.title("Stock Prediction Application")
st.subheader("FBProphet")

stocks = ("AAPL","GOOG","MSFT","^NSEI")
selected_stocks = st.selectbox("Select the stock for prediction: ", stocks)
n_months = st.slider("Months you want predtion for - ",1 , 5)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
# import data_util
yf.pdr_override()
from sklearn.svm import SVC

sns.set(rc={'figure.figsize':(20, 15)})

temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), width=800))

import warnings
warnings.filterwarnings('ignore')

from resources.Evaluation_metrics import *

def plot(df, company):
    p = go.Figure()
    p.add_trace(go.Scatter(x = stocks.index , y = stocks['Open'] , name = 'Stock Open'))
    p.add_trace(go.Scatter(x = stocks.index , y = stocks['Close'] , name = 'Stock Close'))
    p.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible = True)
    st.plotly_chart(p)

@st.cache_data
def stock_data(stock_name, duration, company_name):#duration in years
    end_time = datetime.now()
    start_time = datetime(datetime.now().year - duration, datetime.now().month, datetime.now().day)  #using data of exactly duration year before
    stock=yf.download(stock_name,start_time,end_time)
    stock["comany_name"]=company_name
    return stock

def evaluate_metrics(y_true, y_pred):
    results = {
        'Metric': ['MAPE', 'MAE', 'MSE', 'RMSE', 'R2', 'SMAPE'],
        'Value': [MAPE(y_true, y_pred),
                  MAE(y_true, y_pred),
                  MSE(y_true, y_pred),
                  RMSE(y_true, y_pred),
                  R2(y_true, y_pred),
                  SMAPE(y_true, y_pred)]
    }
    df = pd.DataFrame(results)

    # Add dotted lines
    dotted_style = [dict(selector="th", props=[("border-bottom", "1px dotted #aaaaaa")]),
                    dict(selector="td", props=[("border-bottom", "1px dotted #aaaaaa")])]

    # Apply styling
    styled_df = (df.style
                 .set_properties(**{'text-align': 'center'})
                 .format({'Value': '{:.2f}'})  # Round values to 2 decimal places
                 .set_table_styles(dotted_style)
                 .set_caption('Evaluation Metrics')
                 .set_table_attributes('style="border-collapse: collapse; border: none;"')
                 .set_properties(subset=['Metric'], **{'font-weight': 'bold',}))  # Bold and blue headers

    return styled_df

data_load_state = st.text("Loading Data................")
stocks=stock_data(selected_stocks, 4,selected_stocks)
data_load_state.text("Loaded Data Successfully !")

st.subheader("Latest Stock Price Data")
st.write(stocks.tail())
plot(stocks,selected_stocks)
from prophet import Prophet
from prophet.plot import plot_plotly

# Assuming you have already imported necessary libraries and loaded the data

# Reset index of stocks DataFrame
stocks.reset_index(inplace=True)
# stocks.set_index('Date',inplace=True)
stocks_temp = stocks[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
split=len(stocks_temp)-30 # train it on data from one mont before
future_periods = n_months
stocks_temp_test = stocks_temp.tail(split)
print(split)

stocks_temp_train = stocks_temp.tail(len(stocks_temp) - 30)

FBprophet = Prophet()

FBprophet.fit(stocks_temp_train)
future_periods = 5
future = FBprophet.make_future_dataframe(periods=150)
print("val - ", future_periods)
forecast = FBprophet.predict(future)
st.write("Forecast Data")

st.write(forecast.tail())
fig1 = plot_plotly(FBprophet, forecast)
st.write("Predictions using FBProphet Model")

st.plotly_chart(fig1)
fig1.add_trace(go.Scatter(x=stocks_temp_test['ds'], y=stocks_temp_test['y'], mode='markers', name='Test Data'))

# fig.show()

y_pred=forecast["yhat"].values[(len(stocks_temp)-30):len(stocks_temp)]
y_true=stocks_temp['y'].values[(len(stocks_temp)-30):len(stocks_temp)]
styled_df = evaluate_metrics(y_true, y_pred)
temp = [forecast["yhat"].values[(len(stocks_temp)-30):len(stocks_temp)],stocks_temp['y'].values[(len(stocks_temp)-30):len(stocks_temp)]]
st.write("Model Score for the last 30 day data on different evaluation metrics")
styled_df