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
st.subheader("MultiVariate LSTM")

stocks = ("AAPL","SPY","INFY","^NSEI")
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

def split_data(data,percent_split):
    size=(int((np.shape(data)[0])*(percent_split/100)))
    return data[0:size], data[size:]
def create_dataset(data , step):
    x,y=[],[]
    for i in range((data.shape[0])-step-1):
    
        x.append((data[i:i+step].values.tolist()))
        y.append(data['Close'][data['Close'].index[0]+i+step])
        # y.append(data[i+step]['Close'])


    return np.array(x),np.array(y)


data_load_state = st.text("Loading Data................")
stocks=stock_data(selected_stocks, 4,selected_stocks)
data_load_state.text("Loaded Data Successfully !")

st.subheader("Latest Stock Price Data")
st.write(stocks.tail())
plot(stocks,selected_stocks)



from keras.models import load_model
import io
from sklearn.preprocessing import MinMaxScaler

fit_close = MinMaxScaler(feature_range=(0, 1))

fit_close.fit_transform(np.array(list(stocks['Close'])).reshape(-1,1))

model = load_model("Model/MultiLSTM" + selected_stocks + ".h5")

# Capture the model summary
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
model_summary = "\n".join(model_summary)

# Streamlit app
st.title("LSTM Model Summary")

st.write("## Model Summary")
st.text(model_summary)


cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
step=100
fit = MinMaxScaler(feature_range=(0, 1))
scaled_data = pd.DataFrame()
for col in cols:
    if (col=='Close'):
        scaled_col = fit_close.fit_transform(stocks[col].values.reshape(-1, 1))
        scaled_data[col] = scaled_col.flatten()
    else:
        scaled_col = fit.fit_transform(stocks[col].values.reshape(-1, 1))
        scaled_data[col] = scaled_col.flatten()
train_data,test_data=split_data(scaled_data, 70)
# step = 100
x_train, y_train = create_dataset(train_data, step)
x_test, y_test = create_dataset(test_data, step)
data=np.array(list(stocks['Close'])).reshape(-1,1)

train_predict=model.predict(x_train)
test_predict=model.predict(x_test)    

train_predict=fit_close.inverse_transform(train_predict)
test_predict=fit_close.inverse_transform(test_predict)
y_test=fit_close.inverse_transform(y_test.reshape(-1,1))



testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
# print(testPredictPlot)
testPredictPlot[len(train_predict)+(step*2)+1:len(stocks['Close'])-1] = test_predict

st.title("Stock Price Prediction")

# Improved Matplotlib plot
plt.figure(figsize=(12, 6))

# Plot actual price
plt.plot(
    stocks['Close'].values[:len(train_data) + len(data) - len(test_predict) - len(train_data)],
    color='blue',
    label='Actual Price'
)

# Plot test data price
plt.plot(
    range(len(train_data) + len(data) - len(test_predict) - len(train_data), len(data)),
    stocks['Close'].values[len(train_data) + len(data) - len(test_predict) - len(train_data):],
    color='red',
    label='Test Data Price'
)

# Plot predicted price
plt.plot(
    range(len(data)),
    testPredictPlot,
    color='orange',
    label='Predicted Price'
)

# Add titles and labels
plt.title('Stock Prices Prediction', fontsize=16)
plt.xlabel('Date Index', fontsize=14)
plt.ylabel('Price', fontsize=14)

# Add legend and grid
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Improve the visual aesthetics
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)


styled_df=evaluate_metrics(y_test, test_predict)
st.write("Model Score for the last 30 day data on different evaluation metrics")
styled_df
        