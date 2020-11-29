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

