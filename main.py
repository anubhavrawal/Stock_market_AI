import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers

from datetime import datetime

from csv_reader import csv_to_dataset, past_eval_points

np.random.seed(4)
tf.random.set_seed(4)

# dataset

processed_data_histories, nextday_open_values, unscaled_y, y_normaliser = csv_to_dataset('AAPL_daily.csv')

#90% data for training
split_percent = 0.9 
train_limit = int(processed_data_histories.shape[0] * split_percent)

processed_data_train = processed_data_histories[:train_limit]
y_train = nextday_open_values[:train_limit]

processed_data_test = processed_data_histories[train_limit:]
y_test = nextday_open_values[train_limit:]

unscaled_y_test = unscaled_y[train_limit:]

print("Training data size: ", processed_data_train.shape)
print("Testing data size: " ,processed_data_test.shape)


# Building the Model
lstm_input = Input(shape=(past_eval_points, 5), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)

model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=processed_data_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)


# evaluation

y_test_predicted = model.predict(processed_data_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

y_predicted = model.predict(processed_data_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
#print(scaled_mse)

plt.gcf().set_size_inches(35, 18, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.savefig("sucess.png")


model.save(f'basic_model.h5')
