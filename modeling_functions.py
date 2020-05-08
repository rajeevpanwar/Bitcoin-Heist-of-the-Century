import math
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam


class LSTM_Reshaper(BaseEstimator, TransformerMixin):
    """A class for a custom matrix shape transformation to fit our LSTM model in Scikitkearn's pipelines module. It does not require
    remembering anything from the fit argument, it just outputs a predetermined transform to the shape of the matrix that contains
    the exogenous variables."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.reshape(X.shape[0], 1, X.shape[1])


def keras_model_wrapper(l1_batch_input_shape, l1_neurons, mod_lr):
    """This is a wrapper function that contains the build and compile instructions for our model. It should be passed in as
    an argument within the Tensorflow wrapper class input into the pipeline list."""

    model = Sequential()
    model.add(LSTM(l1_neurons, batch_input_shape=l1_batch_input_shape, stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=mod_lr, beta_1=0.9, beta_2=0.999, amsgrad=False))
    return model

def diff_handle_missing(df, diff, lag=0):
    """Performs a simple difference transformation on a DataFrame, while also dropping the rows at the edges containing incomplete
    data to perform the function."""

    return df.diff(diff).iloc[diff + lag:]

def evaluate_intersection_loss(pred, test):
    """This function calculates the RMSE between the predictions and the test data while ensuring that the evaluation is restricted
    to their overlapping indices."""

    intersection = pred.index.intersection(test.index)
    if intersection.size == 0:
        print("There is no intersection")
        return None
    mse = mean_squared_error(pred.loc[intersection], test.loc[intersection])
    rmse = math.sqrt(mse)
    print(f"MSE: {mse}, RMSE: {rmse})")
    return rmse

def ts_to_supervised(df, lag=1):
    """This will create a target variable to compare against within our RNN network. It does this by offsetting the data to an earlier
    point in time determined by the lag argument. To maintain the same array shape for the model, any new values outside the range of
    actual observations are dropped."""

    df = pd.DataFrame(df)
    idx, shift = df.index, df.index.shift(-lag)
    new_idx = idx.union(shift)
    return df.reindex(new_idx).shift(-lag, fill_value=0).iloc[lag:-lag]

def preprocess_lstm(df, lag=1, diff=1):
    """This chains together several preprocessing functions that must be done before feeding our data to a neural network. It returns
    both the X and y values that can be fed into our model."""

    #Our data requires a log transformation and differencing to achieve stationarity.
    df = np.log(df.copy())
    X = diff_handle_missing(df, diff)

    # We duplicate our observations with an offset to allow for supervised training of our model. Because the input arrays must
    # have a consistent shape we are forced to drop the last observation from our train data.
    y = ts_to_supervised(X, lag)
    X = X[:-lag]
#     y = ts_to_supervised(df, lag)
#     X, y = diff_handle_missing(df, diff), diff_handle_missing(y, diff)
    return X, y

def lstm_predictions(model, train, cols, max_predictions=30):
    """This function takes a trained model along with it's training data to make a set of predictions. They can be capped with the
    'max_predictions' argument."""

    # Instantiates an empty DataFrame to store the predictions.
    predictions = pd.DataFrame([], columns=cols)

    #Takes the undifferenced DataFrame with a natural log transformation applied. It will be used to reconstruct the the undifferenced
    #values from the differenced predictions.
    history = np.log(train[cols].copy())

    # Preprocesses the training data once again. It is out of convenience in reducing the number of input arguments, that this process is
    # called again within the scope of this function.
    X, y = preprocess_lstm(train)
    X, y = X[cols], y[cols]
    scaler = MinMaxScaler()
    scaler.fit(y)

    #This function is called once to compensate for resetting of the model's state at every iteration. On the second call,
    #the predictions are stored into a variable.
    lstm_raw_predictions(model, X)

    #Now that the state has been reinitialized, the predictions can be stored into a variable.
    raw_predictions = lstm_raw_predictions(model, X)
    count = 0
    for raw_prediction in raw_predictions:

        #In order to maintain as much information as possible from the dropped values, this loop needed to behave differently
        #on it's first pass since the omitted actual training data is more valuable than the generated prediction data fpr
        #on the same observation.
        if count <= 1:
            new_index = history.index[-1:]
            prediction = lstm_untransform_value(raw_prediction, scaler, history, 2)
        else:
            new_index = history.index.shift(1)[-1:]
            prediction = lstm_untransform_value(raw_prediction, scaler, history, 1)
        prediction = pd.DataFrame(prediction, index=new_index, columns=cols)

        #If the prediction overlaps with existing data, we make sure to give priority to the actual data when storing the
        #values used to undo the differencing.
        if count <= 1:
            pass
        else:
            history = pd.concat([history, prediction])
        predictions = pd.concat([predictions, np.exp(prediction)])
        count += 1
        if count >= max_predictions:
            break
    return predictions

def lstm_raw_predictions(model, X):
    """Takes a trained model and returns the raw predictions"""

	yhat = model.predict(X)
	return yhat

def lstm_untransform_value(raw_val, scaler, history, offset):
    """This function reverses the preprocessing transformations done on our data to reconstruct the context for the predictions.
    Out of necessitiy in preserving our original observations as much as possible, the value extracted to undo the differencing may
    not be the last observation and is managed by the if statement."""

    reshaped = raw_val.reshape(raw_val.size,1)
    unscaled = scaler.inverse_transform(reshaped)
    if offset == 1:
        undiffed = history.values[-1:] + unscaled
    else:
        undiffed = history.values[-offset:-offset+1] + unscaled
    return undiffed

def pdq_combinations(season_period, max_p, max_d, max_q):
    """Generates a list of combinations for pdq values to be passed into our hyperparameter search function."""

    p, d, q = range(0, max_p+1), range(0, max_d+1), range(0, max_q+1)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], season_period) for x in pdq]
    return pdq, seasonal_pdq

def pdq_hyperparam_search(df, season_period, max_p=1, max_d=1, max_q=1):
    """Generates and searches a list of pdq and season values to find best hyperparameters."""

    pdq, seasonal_pdq = pdq_combinations(season_period, max_p, max_d, max_q)
    all_results=[]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()
                rmse = math.sqrt(results.mse)
                print('ARIMA{}x{}12 - rmse:{}'.format(param, param_seasonal, rmse))
                all_results.append([param, param_seasonal, rmse])
            except:
                continue
    return pd.DataFrame(all_results, columns = ["PDQ", "S-PDQ", "Error"]).sort_values("Error", ascending=True).reset_index(drop=True)

def pdq_hyperparam_df_search(df, pdq_params, new_period=None, n_combs=5):
    """Like pdq_hyperparam search, but iterates through a specific amount of values in a DataFrame. Used to check values if initial data needed resampling
    for speed."""

    all_results=[]
    for i in range(0, n_combs):
        param, param_seasonal = pdq_params.iloc[i:i+1, 0].to_list()[0], pdq_params.iloc[i:i+1, 1].to_list()[0]
        if new_period:
            param_seasonal = list(param_seasonal[0:3])+[new_period]
        try:
            mod = sm.tsa.statespace.SARIMAX(df, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            rmse = math.sqrt(results.mse)
            print('ARIMA{}x{}12 - rmse:{}'.format(param, param_seasonal, rmse))
            all_results.append([param, param_seasonal, rmse])
        except:
            continue
    return pd.DataFrame(all_results, columns = ["PDQ", "S-PDQ", "Error"]).sort_values("Error", ascending=True).reset_index(drop=True)