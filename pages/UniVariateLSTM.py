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
st.subheader("UniVariate LSTM")

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

	
# train_data,test_data=split_data(data, 70)

def create_dataset(data, step):
	X, Y = list(), list()

	for i in range(len(data)-step-1):
		X.append(data[i:(i+step), 0])
		Y.append(data[i + step, 0])
	return np.array(X), np.array(Y)
data_load_state = st.text("Loading Data................")
stocks=stock_data(selected_stocks, 4,selected_stocks)
data_load_state.text("Loaded Data Successfully !")


st.subheader("Latest Stock Price Data")
st.write(stocks.tail())
plot(stocks,selected_stocks)


from keras.models import load_model
import io
from sklearn.preprocessing import MinMaxScaler

fit=MinMaxScaler(feature_range=(0,1))

fit.fit_transform(np.array(list(stocks['Close'])).reshape(-1,1))

model = load_model("Model/UniLSTM" + selected_stocks + ".h5")

# Capture the model summary
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
model_summary = "\n".join(model_summary)

# Streamlit app
st.title("LSTM Model Summary")

st.write("## Model Summary")
st.text(model_summary)

#give summart of model
step=100
data=fit.fit_transform(np.array(list(stocks['Close'])).reshape(-1,1))
train_data,test_data=split_data(data, 70)
# step = 100
unscaled_data=np.array(list(stocks['Close']))
x_train, y_train = create_dataset(train_data, step)
x_test, y_test = create_dataset(test_data, step)
#1 is representing the number of features we are using , here we are doing univariate LSTM . Therefore , 1
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

train_predict=model.predict(x_train)
test_predict=model.predict(x_test)    

train_predict=fit.inverse_transform(train_predict)
test_predict=fit.inverse_transform(test_predict)
y_test=fit.inverse_transform(y_test.reshape(-1,1))
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(step*2)+1:len(data)-1] = test_predict
# Plot the actual data in blue and the predicted data in red
# plt.plot(unscaled_data[:len(train_data)+len(unscaled_data)-len(test_predict)-len(train_data)], color='blue', label='Train Data Price')
# plt.plot(range(len(train_data)+len(unscaled_data)-len(test_predict)-len(train_data),len(unscaled_data)),unscaled_data[len(train_data)+len(unscaled_data)-len(test_predict)-len(train_data):], color='red', label='Test data Price')
# plt.plot(testPredictPlot, color='orange', label='Predicted Price')
# plt.legend()

st.title("Price Prediction using LSTM")

plt.figure(figsize=(12, 6))

plt.plot(
    unscaled_data[:len(train_data) + len(unscaled_data) - len(test_predict) - len(train_data)],
    color='blue',
    label='Train Data Price'
)

plt.plot(
    range(len(train_data) + len(unscaled_data) - len(test_predict) - len(train_data), len(unscaled_data)),
    unscaled_data[len(train_data) + len(unscaled_data) - len(test_predict) - len(train_data):],
    color='red',
    label='Test Data Price'
)


plt.plot(
    range(len(data)),
    testPredictPlot,
    color='orange',
    label='Predicted Price'
)

plt.title('Close Prices Prediction using UniVariate LSTM', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price', fontsize=14)

plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

st.pyplot(plt)

styled_df=evaluate_metrics(y_test, test_predict)
st.write("Model Score for the last 30 day data on different evaluation metrics")
styled_df







