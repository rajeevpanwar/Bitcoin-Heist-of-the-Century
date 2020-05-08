# Module 4 Project

# Project Title - Time Series Forecasting Bitcoin Prices by Corey Hanson & Rajeev Panwar

# Goal
To generate and evaluate various time series forecasting models to predict daily bitcoin prices

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
Our preferred model is ARIMA, but with more tuning and features we would expect to get a better result from the LSTM.

Bitcoin prices are hard to predict. Our best models missed the mark, but there are other exogenous variables that may be good candidates to better explain some of the variability. If we could capture investor sentiment around Bitcoin in more complete manner, that would likely help improve quality of predictions. Several good candidates to generate a better model would be:
* Parsing in the media landscape. Supplementing the model with high quality/extensive samples are taken over time that uses Natural Language Processing to quantify sentiment in the media (both traditional and social)
* Examining on-chain data. The blockchain is unique in that it ties a public ledger with metadata that is transparent to all. Several factors could indicate additional:
    * Supply - This would take form in creating a feature for new coins minted each day by taking the product of the current block reward with the total amount of blocks created on every day in the dataset.
    * Demand - It has been postulated that changes in transaction volume and hashrate could account for some of the utility of a given blockchain, which could indicate demand.

The test data included the last 30 days and has seen a lot of market volatility resulting in higher RMSE than expected. When evaluating it against our testing data, we can see that with the RMSE that the RMSE contained in the earlier observations (215) is significantly lower than the actual RMSE over the entire period (860). Being humble with the time-span, we attempt to predict takes even more importance in the context of a black swan event like the current global COVID-19 pandemic.


# Conclusion
All models struggled to forecast the sudden shift in pricing trajectory due to COVID-19 and as such we were unable to conclude which model performed best given close range of RMSE scores
