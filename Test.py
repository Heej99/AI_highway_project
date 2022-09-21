#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:47:09 2021

@author: deeplearning
"""

#!pip install patchify

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from glob import glob
from PIL import Image
import cv2
import os

import numpy as np
#import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from smooth_tiled_predictions import predict_img_with_smooth_windowing
#from simple_multi_unet_model import jacard_coef  


#import tifffile as tiff
#large_image_stack = tiff.imread('small_dataset_for_training/images/12_training_mito_images.tif')
#large_mask_stack = tiff.imread('small_dataset_for_training/masks/12_training_mito_masks.tif')

#glob returns list of the names which have '.jpg' keyword

cwd = os.getcwd()
print(cwd)

large_image = glob(cwd + '/data/GH050070_31_image.jpg')
large_mask = glob(cwd + '/data/GH050070_31_mask.jpg')

print(large_image)
image = cv2.imread(large_image[0], 1)
mask = cv2.imread(large_mask[0], 1)
mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)


print(image.shape)
end_col = image.shape[1]
print(end_col)
test_image = image[ :, 23565:end_col]
test_mask = mask[ :, 23565:end_col]


print(test_image.shape)

diff = test_image.shape[1] - 512*14
to_col = test_mask.shape[1] - diff

cropped_image = test_image[ :, :to_col]
cropped_mask = test_mask[ :, :to_col]

cv2.imwrite('cropped_test_image.jpg', cropped_image)
cv2.imwrite('cropped_test_mask.jpg', cropped_mask)


'''
cv2.imshow('image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


#patches = patchify(cropped_image, (512,512,3), step = 448) # overlap 64 pixel 512*2 - 64 = 960

import tensorflow as tf
from keras.models import load_model

#Predict on a few images
model = load_model("Unet_simplemodel_test.hdf5", compile=False)

# size of patches
patch_size = 512

# Number of classes 
n_classes = 1

         
#################################################################################
#Predict patch by patch with no smooth blending
###########################################

SIZE_X = (cropped_image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
SIZE_Y = (cropped_mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
large_img = Image.fromarray(cropped_image)
large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
#image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
large_img = np.array(large_img)     

print(large_img.shape)




patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap

print(patches_img.shape)

patches_img = patches_img[:,:,0,:,:,:]

print(patches_img.shape)

patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        
        single_patch_img = patches_img[i,j,:]
        single_patch_img=single_patch_img[:,:,0]
        print(single_patch_img.shape)
        test_img_norm = np.expand_dims(tf.keras.utils.normalize(np.array(single_patch_img), axis=1),2)
        print(test_img_norm.shape)
        test_img_input=np.expand_dims(test_img_norm, 0)
        print(test_img_input.shape)
        
        
        pred = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)
        #pred = np.argmax(pred, axis=3)
        #pred = pred[0, :,:]
                                 
        patched_prediction.append(pred)


patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1], 
                                            patches_img.shape[2], patches_img.shape[3]])

unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

plt.imshow(unpatched_prediction)
cv2.imwrite('test_result.jpg', unpatched_prediction)
plt.axis('off')

