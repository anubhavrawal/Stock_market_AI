import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from datetime import datetime

from csv_reader import csv_to_dataset_2

np.random.seed(7)
tf.random.set_seed(7)
import datetime as dt

# dataset

all_mid_data,train_data_normalized, test_data_normalized, df = csv_to_dataset_2('MSFT_daily.csv')

#Expenontial Moving average

history_point = 50 # Taking past 50 points for making the prediction the next day
N = train_data_normalized.size
date = []

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    running_mean = running_mean*decay + (1.0-decay)*train_data_normalized[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data_normalized[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

plt.figure(figsize = (18,9))

plt.plot(range(df.shape[0]),all_mid_data,color='g',label='Actual')
plt.plot(range(0,N),run_avg_predictions,color='m', label='Prediction')

plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.savefig("sucess3.png")

