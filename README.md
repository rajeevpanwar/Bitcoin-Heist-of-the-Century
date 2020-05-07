# Module 4 Project 

# Project Title - Time Series Forecasting Bitcoin Prices by Corey Hanson & Rajeev Panwar

# Goal - To generate and evaluate best time series forecasting model to predict dailybitcoin prices (upto 15 days into the future)

# Approach 
1. Data Collection - We collected 2562 observations (daily price of bitcoin) going back to 2013-04-28. Our main dataset (https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory) was pulled from Kaggle. It contains data from a selection of various cyptocurrencies where price data was scraped from CoinMarketCap which in the timeline of the crptocurrency market had early on established itself as a popular player in providing market data from this new asset class

2. Data Preprocessing - Depending on the need of the model, we pre-processed the data using date-timestamp, log transformation and differencing to achieve stationarity for model buildout 

3. Choice of Models - We explored 3 different models - ARIMA, Prophet (from Facebook) & a RNN with LTSM 

4. Model Evaluation Metric - RMSE 

# Model Performance 

1. ARIMA - Optimised parameters - p =   , d =   , q =        , RMSE
2. Prophet - baseline prophet model implemented with no hyper-parameter tuning other than to set daily_seasonality to True
3. RNN with LTSM - 


# Findings & Key Learning 
Our preferred model is 

Bitcoin prices are hard to predict. We will probably need to incorporate an exogenous variable to capture investor sentiment around Bitcoin in a better manner to help improve quality of predictions. The test data included last 15-20 days of substiantial market volatility resulting in lower RMSE than expected. Our preferred model seems to do much better when we remove a black swan event like the current global COVID pandemic  from our testing data with a revised RMSE of ___

# Next Steps
Incorporate investor sentiment using count of tweets referencing bitcoin for the same time period to evaluate if predictive power in improved using exogenous variable 


