# Stock Price Prediction

[streamlit-Homepage-2024-07-07-02-07-09.webm](https://github.com/avisharma444/StockPricePrediction/assets/117980764/f49f8faa-e6d0-4cf5-9be2-09351daac8bb)

Website:
- https://stockpricepredictionproject.streamlit.app/
  
Authors:
- Avi Sharma (2022119)
- Baljyot Singh Modi (2022133)

## I. Problem Statement

### Predicting Stock Prices with Machine Learning

The stock market remains an enigma, where predicting stock prices accurately is a formidable challenge. Traditional financial models often struggle with the unpredictability of market shifts and economic trends. In this project, we employ machine learning techniques to unravel these complexities and provide actionable insights for investors.

## II. Datasets

### Yahoo Finance API

We utilized the Yahoo Finance API to access historical stock price datasets for our analysis. Our datasets included:

1. AAPL: Apple Inc. Common Stock
2. NIFTY50: National Stock Exchange's benchmark index
3. SPY: SPDR SP 500 ETF trust
4. INFY: Infosys Limited Stock

These diverse datasets allowed us to test and validate our models across different stocks and market conditions.

## III. Experimental Analysis

### A. Exploratory Data Analysis (EDA)

We performed thorough EDA to understand the datasets:

- Analyzed stock trends and daily returns.
- Examined feature correlations using heatmap visualization.
- Decomposed time series to understand trends and seasonality.
- Utilized rolling and moving averages for predictive insights.

### B. Feature Engineering

Feature Engineering involved creating meaningful features:

- Daily Max Fluctuation and Daily Fluctuation.
- Simple Moving Averages (SMA) for different time periods.
- These features enhanced the predictive power of our models.

### C. Time Series Cross Validation

Due to the time series nature of the data, we implemented Time Series Cross Validation:

- Splitting data into train, test, and validation sets.
- Evaluating model performance across various data splits.

### D. Sequence Generation for LSTM Model

For LSTM models:

- Data scaling using MinMaxScaler.
- Sliding window approach to create input sequences and target values.
- Designed both univariate and multivariate LSTM models.

### E. Stationarity, ADF Test, and ARIMA Modeling

- Ensured stationarity of time series using differencing.
- Conducted Augmented Dickey-Fuller (ADF) Test.
- Applied ARIMA modeling for time series forecasting.

## IV. Methodology

### A. Support Vector Classifiers (SVC)

SVC for classification tasks:

- Used to predict buy/sell signals based on stock price fluctuations.
- Preprocessed data and constructed features for SVC.

### B. Linear Regression

Linear regression model:

- Predicted closing prices using Open, Close, High, and Low features.
- Evaluated using time series cross validation.

### C. XGBoost

XGBoost for gradient boosting:

- Optimized distributed gradient boosting for accurate stock forecasting.
- Utilized in capturing non-linear dependencies in data.

### D. FBProphet

FBProphet for time series forecasting:

- Developed robust models with yearly, weekly, and daily seasonality.
- Handled missing data and shifts in trends effectively.

### E. LSTM Models

#### Univariate LSTM

- Stacked LSTM layers for capturing temporal dependencies.
- Predicted sequential values based on historical data.

#### Multivariate LSTM

- Used multiple time series variables to predict future values.
- Enhanced accuracy by capturing complex relationships.

### F. ARIMA

AutoRegressive Integrated Moving Average model:

- Modeled time series patterns with AR, MA, and differencing.
- Widely used for forecasting in economics and finance.

## V. Results

### A. Support Vector Classifiers (SVC)

- Model Score (R-squared): Reflects the variance explained by the model.

[Insert detailed results and performance metrics here.]

## Conclusion

In conclusion, our project showcases the power of machine learning in predicting stock prices. By leveraging diverse models and comprehensive datasets, we aim to provide valuable insights for investors in navigating the dynamic stock market landscape.
