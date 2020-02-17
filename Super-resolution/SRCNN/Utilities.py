#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:59:30 2020

@author: david
"""
# useful utility functions shared by SRCNN scripts

import h5py
import numpy as np

def read_data_from_file(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1)) # why transpose?
        train_label = np.transpose(label, (0, 2, 3, 1))
        return train_data, train_label
    
    
def read_text_from_file(filename):
    file1 = open(filename,"r+")
    N_epochs = int(file1.read())    
    return(N_epochs)

def write2file(string, filename):
    file1 = open(filename,"w")
    file1.write(string)
    file1.close()
    return