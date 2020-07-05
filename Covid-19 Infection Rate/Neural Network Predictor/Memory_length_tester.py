# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:13:15 2020

@author: David Lloyd
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import time

#%%
# change number of pervious days used as input, see if prediction of next day depends on the number of days used as input ("memory length")

file_location =r"C:/Users/David Lloyd/Documents/Python_Scripts/Covid-19_predictor/"

Infections_data = []
with open(file_location + 'time_series_covid19_confirmed_global.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # skip first row with column headings
    for row in csv_reader:
        if line_count>0:
                
            # deal with uneven list lengths
            if row[-1] == '':
                Short_row = row[4:-1]
                Short_row = [int(x) for x in Short_row]
                Total_infections = np.asarray(np.append(Short_row,0),dtype = 'float')

            else :
                Long_row = row[4:]
                Long_row = [int(x) for x in Long_row]
                Total_infections = np.asarray(Long_row,dtype = 'float')
            # want daily infection number and not total infection number so far
            
            #Daily_infections = np.diff(Total_infections)
            Infections_data.append(Total_infections)
        line_count += 1
        
#%%
np.random.seed(14)
memory_trials = np.arange(2,10,1)

# pick random N_sample_size pairs so that training set size is the same for all
N_sample_size = 6000

Scores_all = []

FS = 18
fig = plt.figure()
plt.rcParams["figure.figsize"] = [8, 8]

begin = time.time()
for memory_length in memory_trials:


    #memory_length = 5
    
    # make dataset first according to current value of memory length   
    # network input is previous "memory length" infection counts
    X = []
    # network output is infection rate on "memory length" + 1 day
    Y = []
    # loop over each row in dataset
    for ii in range(0,len(Infections_data)):
        
        Infections_row = Infections_data[ii] # convert to integer list
        # loop over each entry in row but not all the way to the end
        for jj in range(0,len(Infections_row)-(memory_length+1)):
            # only interested in non-zero infections
            if Infections_row[jj]>0:
                X.append(Infections_row[jj:jj+memory_length])
                Y.append(Infections_row[jj+memory_length])           
    
    pairs = list(zip(X, Y))
    pairs = random.sample(pairs, N_sample_size)  # pick N random pairs
    X_N, Y_N = zip(*pairs)

    X_N = np.asarray(X_N,dtype = 'float')
    Y_N = np.asarray(Y_N,dtype = 'float') 

    X_train, X_test, Y_train, Y_test = train_test_split(X_N, Y_N, test_size=0.2)
    
# train
    initializer = 'glorot_uniform'
    model = Sequential()
    model.add(Dense(32, input_dim=memory_length, activation='relu',kernel_initializer=initializer)) # input_dim = #variables
    # Dense = fully connected, 32 = number of neurons in layer
    #model.add(Dropout(0.0001))
    
    model.add(Dense(16, activation='relu',kernel_initializer=initializer)) # add extra layers (doesn't necessarily improve performance)
    #model.add(Dense(4, activation='linear')) # 8 neurons
    #model.add(Dense(2, activation='relu')) # 8 neurons
    
    model.add(Dense(1, activation='linear',kernel_initializer=initializer))
    
    # Compile model
    checkpoint = ModelCheckpoint("Checkpoint.h5", monitor='acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    ADAM = optimizers.adam(lr=0.00005) #0.00005, 300 - 32-16
    
    callbacks_list = [checkpoint]
    
    model.compile(loss='mean_squared_error', optimizer=ADAM, metrics=['mae'])
    
    # Fit the model (training) [Finding best weigths for prediction]

    history = model.fit(X_train, Y_train, epochs=400, batch_size=64,callbacks=callbacks_list)
    
    scores = model.evaluate(X_test, Y_test) # (loss, metric) best - 43.08
    print('Memory = ' + str(memory_length) + ' days. MAE = ' + str(scores))
 
    Scores_all.append(scores[1])
    

    plt.plot(history.history['loss'], label='train_'+str(memory_length))
    #plt.plot(history.history['val_loss'], label='test')
    plt.yscale('log')
    plt.legend(fontsize = FS)
    plt.xlabel('Epoch',fontsize = FS)
    plt.ylabel('Loss', fontsize = FS)

plt.show()
print('Run time: ' + str(np.round(time.time()-begin)) + ' s')
print(Scores_all)

#%%
import matplotlib.pyplot as plt
FS = 18
fig = plt.figure()
plt.plot(memory_trials, Scores_all,'xr')
