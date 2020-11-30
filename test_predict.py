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

#print(processed_data_test[0:1])
#print( len(processed_data_test[0]) )

print(y_normaliser.inverse_transform(model.predict([ processed_data_test[-2:-1]  , tech_ind_test[-2:-1]])))

y_test_predicted = model.predict([ processed_data_test , tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

y_predicted = model.predict([processed_data_histories, technical_indicators])
y_predicted = y_normaliser.inverse_transform(y_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print("Scaled error: ", scaled_mse)

buy,sell = [],[]
threshold_percent = 0.2

plt.plot(range(0,len(y_predicted)),y_predicted,color='g',label='Actual')
plt.savefig("test.png")

plt.figure(figsize = (18,9))
N = len(unscaled_y_test)

a, = plt.plot(range(0,N),unscaled_y_test,color='g',label='Actual')
b, = plt.plot(range(0,N),y_test_predicted,color='m', label='Prediction')
plt.legend(handles=[a, b])
plt.show()
#plt.plot(range(N,N+ len(y_predicted) ),y_predicted,color='k', label='Prediction2')


plt.savefig("sucess.png")

