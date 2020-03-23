# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:30:30 2020

@author: David Lloyd
"""

# train a neural network to predict the number of infections "tomorrow"
# given the number of infections on the preceeding n days

# Training on cumulative number of cases
# Better to train on daily cases

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
#import matplotlib.pyplot as plt
import csv

# fix the seed to get repeatable results?
np.random.seed(15)

# length of "memory": from how many previous days to take data for input
memory_length = 4

# load training data from .csv files
# first list of n infection numbers
X = []
with open("network_inputs_length_" + str(memory_length) + ".csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # skip first row with column headings
    for row in csv_reader:
        X.append(row)
        
X = np.asarray(X,dtype = 'float')

# then n+1 infection number
Y = []
with open("network_outputs_length_" + str(memory_length) + ".csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # skip first row with column headings
    for row in csv_reader:
        Y.append(row)

Y = np.asarray(Y[0],dtype = 'float')


# define network architecture
# adapted from  #
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# with a few tweaks

model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu')) # input_dim = #variables
# Dense = fully connected, 12 = number of neurons in layer
model.add(Dense(8, activation='relu')) # 8 neurons
model.add(Dense(1, activation='relu')) # signoid for 0<output<1 bounding

# Compile model
ADAM = optimizers.adam(lr=0.0002)

model.compile(loss='mean_squared_error', optimizer=ADAM, metrics=['mae'])

# Fit the model (training) [Finding best weigths for prediction]
model.fit(X, Y, epochs=150, batch_size=50)

scores = model.evaluate(X, Y) # (loss, metric)

#%% test

print(model.predict(np.reshape((2626,3269,3983,5018),(1,memory_length))))
print(model.predict(np.reshape((38,53,64,73),(1,memory_length))))

print(np.shape(np.reshape((2626,3269,3983,5018),(1,memory_length))))
print(np.shape(model.predict(np.reshape((2626,3269,3983,5018),(1,memory_length)))))
#%% recursive generation

initial_data = np.reshape((2626,3269,3983,5018),(1,memory_length))
#initial_data =  np.reshape((38,53,64,73),(1,memory_length))

next_set = np.copy(initial_data)
predictions = []
for ii in range(0,60):
    next_day = model.predict(next_set)
    predictions.append(next_day[0])
    next_set = np.append(next_set[0][1:4],next_day)
    next_set = np.reshape(next_set,(1,memory_length))
    
import matplotlib.pyplot as plt

plt.plot(np.asarray(predictions,dtype = 'float'),'o')
#plt.ylabel('Daily Infections')
