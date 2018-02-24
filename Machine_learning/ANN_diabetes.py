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

# 8 (numeric variable) categories for each (of 768) women 
#1. Number of times pregnant
#2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#3. Diastolic blood pressure (mm Hg)
#4. Triceps skin fold thickness (mm)
#5. 2-Hour serum insulin (mu U/ml)
#6. Body mass index (weight in kg/(height in m)^2)
#7. Diabetes pedigree function
#8. Age (years)
#9. Class variable (0 or 1) 
#  ninth binary category 
# showing 1 for diabetic and 0 for not diabetic.
# split into input (X) and output (Y) variables
X = dataset[:,0:8] #[person,value]
# each element of X is a medical record array size 1x8
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # input_dim = #variables
# Dense = fully connected, 12 = number of neurons in layer
model.add(Dense(8, activation='relu')) # 8 neurons
model.add(Dense(1, activation='sigmoid')) # signoid for 0<output<1 bounding

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model (training) [Finding best weigths for prediction]
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model (applies model to training data to evaluate, usually ...
# seperate "testing" data would be used instead)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
#%%
# create an average patient with mean properties of all 768 real patients 
data_length = len(X[1])
#Ms_average = numpy.arange(data_length) # wrong shape and ends up with rounded
# intergers later on
Ms_average = numpy.ndarray(shape=(1,len(X[1])), dtype=float, order='F')

# Ms average
for x in range(len(X[1])):
#    Ms_average[x] = numpy.mean(X[:,x])
    Ms_average[:,x]=numpy.mean(X[:,x])
#%%
# "create" test patients with increasing glucose concentration, use model to 
# predict their probability of having diabetes

glu_0 = Ms_average[0,1] # baseline glucose level
test_patient = numpy.copy(Ms_average)
L = 500
glu_test = numpy.zeros(L)
p = numpy.copy(glu_test)

for x in range(L):
    glu_test[x] = (1+x/100) *glu_0*0.25 # increment glucose level
    test_patient[0,1] =glu_test[x]
    p[x] = model.predict(numpy.reshape(test_patient,(1,8))) # must be a N,8 ...
# shape array

plt.plot(glu_test/glu_0,p)
plt.ylabel('Diabetes probability')
plt.xlabel('Glucose concentration (x mean value)')

# patient #75 has zero glucose concentration...

#%% As above, but for BMI instead

BMI_0 = Ms_average[0,5]
test_patient2 = numpy.copy(Ms_average)
L = 500
BMI_test = numpy.zeros(L)
p = numpy.copy(BMI_test)

for x in range(L):
    BMI_test[x] = (1+x/100) *BMI_0*0.3 # increment glucose level
    test_patient2[0,5] =BMI_test[x]
    p[x] = model.predict(numpy.reshape(test_patient2,(1,8))) # must be a N,8 ...
# shape array

plt.plot(BMI_test/BMI_0,p)
plt.ylabel('Diabetes probability')
plt.xlabel('BMI (x mean value)')
