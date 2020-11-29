import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from keras.models import load_model
from keras.utils import plot_model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from datetime import datetime

from csv_reader import csv_to_dataset, past_eval_points



processed_data_histories, technical_indicators, nextday_open_values, unscaled_y, y_normaliser  = csv_to_dataset('AAPL_daily.csv')

#90% data for training
split_percent = 0.9 
train_limit = int(processed_data_histories.shape[0] * split_percent)

processed_data_train = processed_data_histories[:train_limit]
tech_ind_train = technical_indicators[:train_limit]

y_train = nextday_open_values[:train_limit]

processed_data_test = processed_data_histories[train_limit:]
tech_ind_test = technical_indicators[train_limit:]
y_test = nextday_open_values[train_limit:]

unscaled_y_test = unscaled_y[train_limit:]

model = load_model('basic_model.h5')

buy,sell = [],[]
threshold_percent = 0.2

x=0

#model.predict( [[data], [indicator]]

print(processed_data_test.shape)
new_test = processed_data_test[0][0]
print(processed_data_test[1].shape)

print()

print(tech_ind_test.shape)
print(tech_ind_test[1].shape)

predict_val = np.expand_dims(processed_data_test[0], -1)

predict_indicator = np.expand_dims(tech_ind_test[0], -1)

model.predict( [[predict_val], [predict_indicator]] )



#new_val = model.predict( [processed_data_test[0],tech_ind_test[0]] )

#print(new_val)


# for data, indicator in zip(processed_data_test,tech_ind_test):
#     today_price_normalized = np.array([[ data[-1][0] ]])
#     today_price = y_normaliser.inverse_transform(today_price_normalized)
#     predicted_nextDay_price = np.squeeze(y_normaliser.inverse_transform( model.predict( [[data], [indicator]] ) ))
#     delta = predicted_nextDay_price - today_price

#     if delta > threshold_percent:
#         buy.append((x,today_price[0][0]))
    
#     elif delta < -threshold_percent:
#         sell.append ((x,today_price[0][0]))
    
#     x = x+1

# print(buy)
# print(sell)