import pandas as pd
from sklearn import preprocessing
import numpy as np
import datetime as dt

from pandas_datareader import data

import matplotlib.pyplot as plt

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

    #Adding more complexity to the model
    technical_indicators = []
    for hist_val in data_histories_normalised:
        # taking SMA of closing price by hist_val[3]
        sma = np.mean(hist_val[:, 3]) #simple moving average
        technical_indicators.append(np.array([sma]))
        
    technical_indicators = np.array(technical_indicators)

    scalar_tech_ind = preprocessing.MinMaxScaler()
    technical_indicators_normalised = scalar_tech_ind.fit_transform(technical_indicators)

    #Checking if dim of x is equals to the dim of y
    assert data_histories_normalised.shape[0] == nextday_open_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    #Return the values
    return data_histories_normalised, technical_indicators_normalised, nextday_open_values_normalised, nextday_open_values, y_normaliser

#csv_to_dataset('AAPL_daily.csv')

def csv_to_dataset_2(csv_path):
    df = pd.read_csv(csv_path)
    df = pd.DataFrame(df.values[::-1], df.index, df.columns)
    
    '''
    plt.figure(figsize = (30,18))
    plt.plot(range(df.shape[0]) , (df['Low']+df['High'])/2.0)
    plt.xticks(range(0,df.shape[0],300),df['date'].loc[::300],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.savefig('mid_price.png')
    '''

    high = df.loc[:,'High'].values 
    low= df.loc[:,'Low'].values
    mid_prices= (high+low)/2.0

    #90% data for training
    split_percent = 0.9 
    train_limit = int(len(mid_prices) * split_percent)

    train_data = mid_prices[:train_limit]
    test_data = mid_prices[train_limit:]

    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

    train_data_normaliser = preprocessing.MinMaxScaler() #0->1 DAta normalization
    train_data_normalized = train_data_normaliser.fit_transform(train_data)

    test_data_normaliser = preprocessing.MinMaxScaler() 
    test_data_normalized = test_data_normaliser.fit_transform(test_data)
    
    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data_normalized,test_data_normalized],axis=0) 

    return all_mid_data,train_data_normalized, test_data_normalized, df



#csv_to_dataset_2('MSFT_daily.csv')