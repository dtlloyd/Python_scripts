# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:30:30 2020

@author: David Lloyd
"""

# train a neural network to predict the number of infections "tomorrow"
# given the number of infections on the preceeding n days

# Training on cumulative number of cases
# Better to train on daily cases?

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv
import time
#%%
np.random.seed(15)
# length of "memory": from how many previous days to take data for input
memory_length = 5

# load training data from .csv files
# first list of n infection numbers
X = []
with open("Total_infected_inputs_length_" + str(memory_length) + ".csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # skip first row with column headings
    for row in csv_reader:
        X.append(row)
        
X = np.asarray(X,dtype = 'float')

# then n+1 infection number
Y = []
with open("Total_infected_outputs_length_" + str(memory_length) + ".csv")as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # skip first row with column headings
    for row in csv_reader:
        Y.append(row)

Y = np.asarray(Y[0],dtype = 'float')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#%%
# fix the seed to get repeatable results?
np.random.seed(15)

model = Sequential()
initializer = 'glorot_uniform'
model.add(Dense(32, input_dim=memory_length, activation='relu',kernel_initializer=initializer)) # input_dim = #variables
# Dense = fully connected, 12 = number of neurons in layer
#model.add(Dropout(0.00001))

model.add(Dense(16, activation='relu',kernel_initializer=initializer)) # 8 neurons
#model.add(Dense(4, activation='linear')) # 8 neurons
#model.add(Dense(2, activation='relu')) # 8 neurons

model.add(Dense(1, activation='relu',kernel_initializer=initializer))

# Compile model
checkpoint = ModelCheckpoint("Checkpoint.h5", monitor='acc', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='min')
ADAM = optimizers.adam(lr=0.00005) #0.00005, 300 - 32-16

callbacks_list = [checkpoint]

model.compile(loss='mean_squared_error', optimizer=ADAM, metrics=['mae'])

# Fit the model (training) [Finding best weigths for prediction]
begin = time.time()
history = model.fit(X_train, Y_train, epochs=350, batch_size=64,callbacks=callbacks_list)

scores = model.evaluate(X_test, Y_test) # (loss, metric) best - 43.08
print(scores)
print('Training time: ' + str(np.round(time.time()-begin)) + ' s')

#%%

FS = 18
fig = plt.figure()
plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
plt.yscale('log')
plt.legend(fontsize = FS)
plt.xlabel('Epoch',fontsize = FS)
plt.ylabel('Loss', fontsize = FS)
plt.show()

#%% test

data_vec = (22141., 25150., 29474., 33718., 38168.)
#data_vec = (139.,149., 151., 156., 169.)
initial_data = np.reshape(data_vec,(1,memory_length))
#initial_data = np.reshape((5683.,6650.,8077.,9529.,11658.),(1,memory_length,1))
plus_one = model.predict(initial_data)
print(plus_one)
#%% recursive generation

next_set = np.copy(initial_data)
predictions = []
extra_days = 14
for ii in range(0,extra_days):
    next_day = model.predict(next_set)
    predictions.append(next_day[0])
    next_set = np.append(next_set[0][1:memory_length],next_day)
    next_set = np.reshape(next_set,(1,memory_length))
    
import matplotlib.pyplot as plt

plt.plot(np.arange(0,memory_length,1),data_vec,'x')
plt.plot(np.arange(memory_length,extra_days+memory_length,1),np.asarray(predictions,dtype = 'float'))
#plt.ylabel('Daily Infections')

#%% compare performance to simple fit to benchmark mean absolute error
# check types, shapes and source of warnings
def simple_expo(x,a,b,c):
    
    return a * np.exp(-b * x) + c

def simple_lin(x,m,c):
    return m * x + c


xx = np.arange(0,memory_length,1)
xx = [float(ii) for ii in xx]

MAE_NN = []
MAE_FIT = []
for ii in range(0,len(Y_test)):
    
    test_ind = ii
    test_vec = np.copy(X_test[test_ind])
    
    
    
    try:
        p_exp, pc_exp = curve_fit(simple_expo, xx, test_vec, \
                                  p0 = (test_vec[0],\
                                        (test_vec[-1]-test_vec[0])/test_vec[-1],1), \
                                        maxfev = 4000)
        
        #plt.plot(xx,np.asarray(test_vec),'o') 
        #plt.plot(xx,simple_expo(xx,*p_exp))
        
        initial_data = np.reshape(test_vec,(1,memory_length))
        #initial_data = np.reshape((5683.,6650.,8077.,9529.,11658.),(1,memory_length,1))
        plus_one = model.predict(initial_data)
        
        #plus_one = np.round(plus_one)
 
        MAE_FIT.append(np.abs(simple_expo(np.amax(xx)+1,*p_exp)-Y_test[test_ind]))
        MAE_NN.append(np.abs(plus_one-Y_test[test_ind]))
    except:
        pass
print(np.mean(MAE_FIT))
print(np.mean(MAE_NN))
#data_vec = (139.,149., 151., 156., 169.)

#%%
plt.plot(MAE_FIT,'x')
plt.plot(np.reshape(MAE_NN,(len(MAE_NN),1)),'o')
plt.yscale('log')
plt.xlim(0,50)
plt.ylim(1,1000)