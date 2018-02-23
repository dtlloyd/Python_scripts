# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:12:51 2018

@author: lloyd
"""
# The bulk of this script was taken from Jason Brownlee's website:
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.pyplot as plt
# fix random seed for reproducibility
numpy.random.seed(6)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
#%%
# create an average patient with mean properties of all 768 real patients 
data_length = len(X[1])
Ms_average = numpy.arange(data_length)

# Ms average
for x in range(len(X[1])):
    Ms_average[x] = numpy.mean(X[:,x])

#%%
# "create" test patients with increasing glucose concentration, use model to 
# predict their probability of having diabetes

glu_0 = Ms_average[1] # baseline glucose level
test_patient = numpy.copy(Ms_average)
L = 500
glu_test = numpy.zeros(L)
p = numpy.copy(glu_test)

for x in range(L):
    glu_test[x] = (1+x/100) *glu_0*0.3 # increment glucose level
    test_patient[1] =glu_test[x]
    p[x] = model.predict(numpy.reshape(test_patient,(1,8))) # must be a N,8 ...
# shape array

plt.plot(glu_test/glu_0,p)
plt.ylabel('Diabetes probability')
plt.xlabel('Glucose concentration (x mean value)')

# patient #75 has zero glucose concentration...