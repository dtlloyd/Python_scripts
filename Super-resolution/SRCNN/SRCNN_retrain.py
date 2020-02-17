#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Tue Feb 11 13:02:41 2020

@author: david
"""
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import sys
import os
import matplotlib.pyplot as plt

from Utilities import read_data_from_file, read_text_from_file, write2file

def continue_training(n_epochs):
    data, label = read_data_from_file("./train.h5")
    val_data, val_label = read_data_from_file("./test.h5")
    
    new_model = load_model("SRCNN_check.h5")
    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
        # try to get tensorboard working

    cwd = os.getcwd() # current working directory
    #log_dir=cwd + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = cwd + "/logs/fit/" +'logfile'
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)

    callbacks_list = [tensorboard_callback, checkpoint]
    
    old_epochs = read_text_from_file('Epoch_number.txt')
    
    history = new_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs=n_epochs,\
                        initial_epoch = old_epochs, verbose=0)
    
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    new_model.save('Trained_mod.h5')
    
    write2file(str(n_epochs),'Epoch_number.txt')
    
if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    continue_training(n_epochs)