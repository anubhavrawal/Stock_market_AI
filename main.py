import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from datetime import datetime

from csv_reader import csv_to_dataset, past_eval_points

np.random.seed(7)
tf.random.set_seed(7)

# dataset

ticker_name = 'TSLA'

processed_data_histories, technical_indicators, nextday_open_values, unscaled_y, y_normaliser  = csv_to_dataset('%s_daily.csv'%ticker_name)

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

print("Training data size: ", processed_data_train.shape)
print("Testing data size: " ,processed_data_test.shape)


# Building the Model
lstm_input = Input(shape=(past_eval_points, 5), name='lstm_input')
dense_layering_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

#Branch 1 working on [`lstm_input`]
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
lstm_branch = Model(inputs = lstm_input, outputs=x) #branching  for later use

#Branch 2 working on [`dense_layering_input`]
x_1 = Dense(20, name='tech_dense_0')(dense_layering_input)
x_1 = Activation("relu", name='tech_relu_0')(x_1)
x_1 = Dropout(0.2, name='tech_dropout_0')(x_1)
technical_indicators_branch = Model(inputs=dense_layering_input, outputs=x_1)

merged_branch = concatenate([lstm_branch.output, technical_indicators_branch.output], name = 'concatenate')

#x = Dense(64, name='dense_0')(x)
#x = Activation('sigmoid', name='sigmoid_0')(x)
#x = Dense(1, name='dense_1')(x)
#output = Activation('linear', name='linear_output')(x)

y = Dense(64, activation="sigmoid", name='dense_pooling')(merged_branch)
y = Dense(1, activation="linear", name='dense_out')(y)

#
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=y)
#model = Model(inputs=lstm_input, outputs=output)

adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
#plot_model(model, to_file='model.png', show_shapes=True)

#Model Training
#model.fit(x=processed_data_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1) #Simple
model.fit(x=[processed_data_train,tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1) # Complex

# evaluation

#y_test_predicted = model.predict(processed_data_test)

y_test_predicted = model.predict([processed_data_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

y_predicted = model.predict([processed_data_histories, technical_indicators])
y_predicted = y_normaliser.inverse_transform(y_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print("Scaled error: ", scaled_mse)

plt.plot(range(0,len(y_predicted)),y_predicted,color='g',label='Actual')
plt.savefig("test.png")

plt.figure(figsize = (18,9))
N = len(unscaled_y_test)

a, = plt.plot(range(0,N),unscaled_y_test,color='g',label='Actual')
b, = plt.plot(range(0,N),y_test_predicted,color='m', label='Prediction')
plt.legend(handles=[a, b])

plt.savefig("{0}_{1:.2f}_sucess.png".format(ticker_name,scaled_mse))


model.save('models/%s_model.h5' %ticker_name)
