# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 09:52:17 2018

@author: David Lloyd
"""
# Script not working
# network parameter defintion - fine
# network training - fine
# Falls over at line 100 with error:
# ValueError: Error when checking : expected conv2d_1_input to have 4 
#dimensions, but input is array with shape (3, 32, 32)
# preict command works if reshapen to size (1,3,32,32)

#%%
# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# create a grid of 3x3 images
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(toimage(X_train[i]))
    # or image.fromarray instead?
# show the plot
pyplot.show()
#%%

# Simple CNN model for CIFAR-10
import numpy
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

t0 = time.time()
#
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# reduced training data size (<50K) to speed things up a little

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32') # convert to float for easier manipulation
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# size(50000,3,32,32) 50k 32x32 rgb images
#
# one hot encode outputs: binary string with one "1" and rest "0"
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
#
# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 4 # probably needs to be >250; one epoch ~15 mins on laptop...
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
#
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

t1 = time.time()
total = t1-t0

#%%
from PIL import Image
import numpy
from matplotlib import pyplot

filename = 'Bertu_box2.png'
im = Image.open(filename)
# im.show()# opens in windows picture viewer
#pyplot.imshow(toimage(im)) # opens in console or Image.fromarray
upscale_pic = numpy.array(im)
upscale_pic = upscale_pic[:,:,0:3] # why four layers

#int_list = range(10)
#float_list = [float(i) for i in int_list]
 
L = 18;
list_32s = numpy.zeros(L)
for i in range(L):
    list_32s[i] = 32*(1+i)
    
# find multiple of 32 closest to but less than dimension of imported image
targ = len(upscale_pic) # dimension of imported image
list1 = list_32s-targ
list1[list1 > 0] = 33

index1 = numpy.unravel_index(numpy.argmin(abs(list1), axis=None), abs(list1).shape)
index2 =  int(list_32s[index1])
upscale_pic = upscale_pic[0:index2,0:index2] # crop to multiple of 32

downscale_pic = numpy.ndarray([32,32,3])
down_fact = int(index1[0]+1) # downscaling factor

for i in range(32):
    x_ind = range(i*down_fact,i*down_fact+down_fact)
    for j in range (32):
        y_ind = range(j*down_fact,j*down_fact+down_fact)
        for k in range(3):
            downscale_pic[i,j,k] = numpy.mean(upscale_pic[x_ind,y_ind,k])

pyplot.imshow(numpy.sum(downscale_pic,2))
downscale_pic =numpy.transpose(downscale_pic,(2,0,1))     
#img = Image.fromarray(upscale_pic, 'RGB')
#pyplot.imshow(toimage(img))
#pyplot.imshow(numpy.sum(downscale_pic,2))

#img_r = Image.fromarray(downscale_red, 'RGB')# something going wrong here.
#pyplot.imshow(img_r)
#%%
# reshape to 1,3,32,32
downscale_pic = numpy.reshape(downscale_pic,(1,3,32,32))
model.predict(downscale_pic)
# %%
downscale_red = numpy.copy(downscale_pic)
downscale_red[:,:,1] = 0            
red_chan = numpy.copy(upscale_pic)
red_chan[:,:,1] = 0
red_chan[:,:,2] = 0
red_img = Image.fromarray(red_chan, 'RGB')
pyplot.imshow(toimage(red_img))

downscale_red = numpy.ndarray([32,32,3])

for i in range(32):
    for j in range (32):
        for k in range(3):
            downscale_red[i,j,k] = red_chan[i,j,k]
red_img_reduced = Image.fromarray(downscale_red, 'RGB')
pyplot.imshow(toimage(red_img_reduced))