import pandas as pd
from sklearn import preprocessing
import numpy as np

#The model will look back past 50 sets of data (day currently as we have daily values) to predict the next day's price
past_eval_points = 50


def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1) # lets drop date as we only need time series to train the model
    data = data.drop(0, axis=0) # remove the reminants

    data = data.values # lets ditch the csv format and just keep the values

    #For convergence its better that the data would be in a range of 0-1 
    #As sklearn already has it why bother making one....
    data_normaliser = preprocessing.MinMaxScaler() 
    data_normalised = data_normaliser.fit_transform(data)

    #data_histories_normalised
    # using the last {past_eval_points} open close high low volume data points, predict the next open value
    data_histories_normalised = np.array([data_normalised[i:i + past_eval_points].copy() for i in range(len(data_normalised) - past_eval_points)]) # Our x-axis
    nextday_open_values_normalised = np.array([data_normalised[:, 0][i + past_eval_points].copy() for i in range(len(data_normalised) - past_eval_points)])
    nextday_open_values_normalised = np.expand_dims(nextday_open_values_normalised, -1)

    #y-values with respect of x
    nextday_open_values = np.array([data[:, 0][i + past_eval_points].copy() for i in range(len(data) - past_eval_points)])
    nextday_open_values = np.expand_dims(nextday_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(nextday_open_values)
    
    #Checking if dim of x is equals to the dim of y
    assert data_histories_normalised.shape[0] == nextday_open_values_normalised.shape[0]
    #Return the values
    return data_histories_normalised, nextday_open_values_normalised, nextday_open_values, y_normaliser

#csv_to_dataset('AAPL_daily.csv')