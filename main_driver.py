import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from datetime import datetime

from DataGenerator import DataGeneratorSeq

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

decay = (2/51)

for pred_idx in range(1,N):
    #running_mean = running_mean*decay + (1.0-decay)*train_data_normalized[pred_idx-1]
    running_mean = (train_data_normalized[pred_idx] * decay) + train_data_normalized[pred_idx-1] * (1.0-decay)
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
plt.savefig("sucess2.png")


dg = DataGeneratorSeq(train_data_normalized,5,5)
u_data, u_labels = dg.unroll_batches()

""" feed_dict = {}
for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
    feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
    feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)

print(feed_dict) """



'''
D = 1 # Dimensionality of the data. Since your data is 1-D this would be 1
num_unrollings = 50 # Number of time steps you look into the future.
batch_size = 500 # Number of samples in a batch
num_nodes = [200,200,150] # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes) # number of layers
dropout = 0.2 # dropout amount

epochs = 30
valid_summary = 1 # Interval you make test predictions

n_predict_once = 50 # Number of steps you continously predict for


'''

# Building the Model
""" lstm_input = Input(shape=(500, 1), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)

model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=u_data, y=u_labels, batch_size=500, epochs=30, shuffle=True, validation_split=0.1)

n_future = 50
our_predictions = []

test_points_seq = np.arange(11000,12000,50).tolist()


for pred_i in range(n_future):
    pred = model.predict() """



