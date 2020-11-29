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