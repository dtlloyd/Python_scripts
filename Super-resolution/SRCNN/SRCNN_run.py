#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:36:09 2020

@author: david
"""
# script to be run from the command line to train and then test SRCNN using
# Keras

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
import sys
import os
import matplotlib.pyplot as plt

from Utilities import read_data_from_file, write2file
    
def model():
    
    # define model type
    SRCNN = Sequential()
    
    pad_size = 8 # can cropping and padding to eliminate border effects?
    # (9,5,5) architecture 
    # add padding
    SRCNN.add(ZeroPadding2D(padding=(pad_size, pad_size)))
    
    SRCNN.add(Conv2D(filters=64, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    
    #SRCNN.add(ZeroPadding2D(padding=(pad_size, pad_size))) # why more padding needed?
    
    SRCNN.add(Conv2D(filters=32, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True))
   # SRCNN.add(ZeroPadding2D(padding=(pad_size, pad_size)))
    
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    # remove padding
    SRCNN.add(Cropping2D(cropping = (pad_size-8,pad_size-8))) # this works
    # but doesn't seem right. Where does the 8 come from?
    
    # define optimizer
    adam = Adam(lr=0.0001) # default val: 0.0003
    
    # compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return SRCNN
    
    
def train2(n_epochs):
    srcnn_model = model()
    
    #print(srcnn_model.summary())
    
    #load data
    data, label = read_data_from_file("./train.h5")
    val_data, val_label = read_data_from_file("./test.h5")

    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    
    # Does not run with tensorboard callback. leave it out for now
    cwd = os.getcwd() # current working directory
    #log_dir=cwd + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = cwd + "/logs/fit/" +'logfile'
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)
    
    callbacks_list = [tensorboard_callback, checkpoint]
    
    history = srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs=n_epochs, verbose=0)
    #print(history.history)
    

    #fig = plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    srcnn_model.save('Trained_mod.h5')
    
    write2file(str(n_epochs),'Epoch_number.txt')
    
    
    return

if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    train2(n_epochs)
    