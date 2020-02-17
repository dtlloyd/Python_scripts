#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:04:05 2020

@author: david
"""
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from Utilities import read_data_from_file
from skimage.measure import compare_ssim as SSIM_raw

#
def SSIM(array1,array2): 
    return SSIM_raw(array1,array2)

def RMSE(array1,array2):
    return np.sqrt(np.mean((array1-array2)**2))

def single_shave(array,length):
    return array[length:-length,length:-length]

def bulk_shave(data,length):
    return data[:,length:-length,length-length,:]

K1_B10 = 774.8853
K2_B10 = 1321.0789
ML_B10 = 3.3420E-04
AL_B10 = 0.10000

def rad2temp(array):
    return (K2_B10/np.log(K1_B10/(array*ML_B10 + AL_B10)+1))
#

trained_mod = load_model('Trained_mod.h5')
test_images_LR,test_images_HR = read_data_from_file('test.h5') # import test images
test_images_SR = trained_mod.predict(test_images_LR,batch_size=1)

#
L_shave = 8
#print('First RMSE in Radiance:')
#print('RMSE bicubic: ' + str(RMSE(test_images_HR,test_images_LR)))
#print('RMSE super-res: ' + str(RMSE(test_images_HR,test_images_SR)))

#print('RMSE bicubic (cropped): ' +  str(RMSE(bulk_shave(test_images_HR,L_shave),bulk_shave(test_images_LR,L_shave))))
#print('RMSE super-res: ' + str(RMSE(bulk_shave(test_images_HR,L_shave),bulk_shave(test_images_SR,L_shave))))
#print('')
print('RMSE in Kelvin:')
print('RMSE bicubic: ' + str(RMSE(rad2temp(test_images_HR),rad2temp(test_images_LR))))
print('RMSE super-res: ' + str(RMSE(rad2temp(test_images_HR),rad2temp(test_images_SR))))

print('RMSE bicubic (cropped): ' +  \
      str(RMSE(rad2temp(bulk_shave(test_images_HR,L_shave)),rad2temp(bulk_shave(test_images_LR,L_shave)))))
print('RMSE super-res: ' + str(RMSE(rad2temp(bulk_shave(test_images_HR,L_shave)),rad2temp(bulk_shave(test_images_SR,L_shave)))))

#
SSIM_all_SR = []
SSIM_all_bic = []
SSIM_all_SR_crop = []
SSIM_all_bic_crop = []


for ii in range(0,np.shape(test_images_LR)[0]):
    SSIM_all_SR.append(SSIM(rad2temp(test_images_HR[ii,:,:,0]),rad2temp(test_images_SR[ii,:,:,0])))
    SSIM_all_bic.append(SSIM(rad2temp(test_images_HR[ii,:,:,0]),rad2temp(test_images_LR[ii,:,:,0])))
    
    SSIM_all_SR_crop.append(SSIM(rad2temp(single_shave(test_images_HR[ii,:,:,0],L_shave))\
                                 ,rad2temp(single_shave(test_images_SR[ii,:,:,0],L_shave))))
    
    SSIM_all_bic_crop.append(SSIM(rad2temp(single_shave(test_images_HR[ii,:,:,0],L_shave))\
                                 ,rad2temp(single_shave(test_images_LR[ii,:,:,0],L_shave))))
print('')
print('SSIM')
print('SSIM bicubic: ' + str(np.mean(SSIM_all_bic)))
print('SSIM super-res: ' + str(np.mean(SSIM_all_SR)))
print('SSIM bicubic (cropped): ' + str(np.mean(SSIM_all_bic_crop)))
print('SSIM super-res (cropped): ' + str(np.mean(SSIM_all_SR_crop)))


#
test_ind = 78
shave_ind = 8

FS = 18
plt.rcParams['figure.figsize'] = [12, 8]
plt.subplot(2,3,1)
plt.imshow(rad2temp(test_images_LR[test_ind,:,:,0]),cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Low Res. (bicubic interp.)')

plt.subplot(2,3,2)
plt.imshow(rad2temp(test_images_HR[test_ind,:,:,0]),cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('High Res.')


plt.subplot(2,3,3)
plt.imshow(rad2temp(test_images_SR[test_ind,:,:,0]),cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Super Res.')

plt.subplot(2,3,4)
plt.imshow(single_shave(rad2temp(test_images_LR[test_ind,:,:,0]),shave_ind),cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Low Res. (bicubic interp.)')

plt.subplot(2,3,5)
plt.imshow(single_shave(rad2temp(test_images_HR[test_ind,:,:,0]),shave_ind),cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('High Res.')


plt.subplot(2,3,6)
plt.imshow(single_shave(rad2temp(test_images_SR[test_ind,:,:,0]),shave_ind),cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Super Res.')

plt.show()