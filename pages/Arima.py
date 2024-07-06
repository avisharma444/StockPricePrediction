import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
yf.pdr_override()
from datetime import datetime
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
# import data_util
# import data_util
yf.pdr_override()
from sklearn.svm import SVC

from prophet.plot import plot_plotly


sns.set(rc={'figure.figsize':(20, 15)})

temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), width=800))

import warnings
warnings.filterwarnings('ignore')

from resources.Evaluation_metrics import *
st.title("Stock Prediction Application")
st.subheader("Autoregressive Integrated Moving Average(ARIMA) ")

stocks = ("AAPL","GOOG","MSFT","SPY")
selected_stocks = st.selectbox("Select the stock for prediction: ", stocks)


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
stocks=stock_data(selected_stocks, 2,selected_stocks)
data_load_state.text("Loaded Data Successfully !")


st.subheader("Latest Stock Price Data")
st.write(stocks.tail())
plot(stocks,selected_stocks)


stocks = stocks[['Close']].copy()
stocks['FirstDiff'] = stocks['Close'].diff().dropna()

acf_vals = acf(stocks['FirstDiff'][1:], fft=False)
acf_plot = go.Figure(data=[
    go.Bar(x=list(range(1, len(acf_vals))), y=acf_vals[1:], marker_color='blue')
])
acf_plot.update_layout(
    title='Auto-Correlation Function (ACF) Bar Chart',
    xaxis_title='Lag',
    yaxis_title='ACF Value',
    xaxis=dict(tickmode='linear')
)
pacf_vals = pacf(stocks['FirstDiff'][1:])
pacf_plot = go.Figure(data=[
    go.Bar(x=list(range(1, len(pacf_vals))), y=pacf_vals[1:], marker_color='red')
])
pacf_plot.update_layout(
    title='Partial Auto-Correlation Function (PACF) Bar Chart',
    xaxis_title='Lag',
    yaxis_title='PACF Value',
    xaxis=dict(tickmode='linear')
)


st.plotly_chart(acf_plot)
st.plotly_chart(pacf_plot)

train=pd.DataFrame(stocks['Close'][0:int(len(stocks)*0.70)])
test=pd.DataFrame(stocks['Close'][int(len(stocks)*0.70):])

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()

# Display the model summary
model_summary = model_fit.summary()

# Streamlit app
st.title("ARIMA Model Summary")

st.write("## Model Summary")
st.text(model_summary)

residuals = pd.DataFrame(model_fit.resid[1:])

# Create the residual plot
residual_plot = go.Figure(data=[
    go.Scatter(x=residuals.index, y=residuals[0], mode='lines', name='Residuals')
])
residual_plot.update_layout(
    title='Residuals Plot',
    xaxis_title='Index',
    yaxis_title='Residuals'
)

# Create the histogram of residuals
histogram = go.Figure(data=[
    go.Histogram(x=residuals[0], histnorm='probability', name='Residuals')
])
histogram.update_layout(
    title='Histogram of Residuals',
    xaxis_title='Residuals',
    yaxis_title='Density'
)

# Streamlit app
st.title("ARIMA Model Residual Analysis")

st.write("## Residual Plot")
st.plotly_chart(residual_plot)

st.write("## Histogram of Residuals")
st.plotly_chart(histogram)

import warnings
warnings.filterwarnings("ignore")
rolling=int(len(stocks)*0.70)
train=pd.DataFrame(stocks['Close'][0:rolling])
test=pd.DataFrame(stocks['Close'][rolling:])
predictions_rolling=[]
for i in range(len(test)):

    model = ARIMA(train, order=(2,1,2))
    model_fit = model.fit()
    pred=model_fit.forecast()
    predictions_rolling.append(pred)
    # print(float(pred),test['Close'][i])
    rolling+=1
    train=pd.DataFrame(stocks['Close'][0:rolling])
# print(predictions_rolling)


rolling_predicted_df = pd.DataFrame({'Predicted_Value': predictions_rolling}, index=test.index)

# Plotting with Matplotlib (for comparison)
plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(train.index, train, color='blue', label='Training Data')

# Plot test data
plt.plot(test.index, test, color='red', label='Test Data')

# Plot predicted data
plt.plot(rolling_predicted_df.index, rolling_predicted_df['Predicted_Value'], color='orange', label='Predicted Data')

plt.title('Close Prices Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Close Price Difference')
plt.legend()
plt.grid(True)

# Streamlit app
st.title("ARIMA Model Predictions")
st.pyplot(plt)
