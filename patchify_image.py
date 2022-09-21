# pip install patchify


import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from glob import glob
from PIL import Image
import cv2
import os
#import tifffile as tiff
#large_image_stack = tiff.imread('small_dataset_for_training/images/12_training_mito_images.tif')
#large_mask_stack = tiff.imread('small_dataset_for_training/masks/12_training_mito_masks.tif')


#glob returns list of the names which have '.jpg' keyword

print(os.getcwd())


cwd = os.getcwd()

large_image_stack = glob(cwd + '/data/original/image/*.jpg')
large_mask_stack = glob(cwd + '/data/original/mask/*.jpg')


print(len(large_image_stack))

'''
image = cv2.imread(large_image_stack[0],1)

patches = patchify(image, (512,512,3), step = 448) # overlap 64 pixel 512*2 - 64 = 960


print(patches.shape)
# reconstruct_image = unpatchify(patches, image.shape)

#cv2.imwrite('./image/image.jpg', single_img)

cv2.imshow('image', patches[0,1,0])
cv2.waitKey(0)
cv2.destroyAllWindows()



#image = Image.open(large_image_stack[0])
#plt.imshow(image)

'''

#print(large_image_stack[0].split('/')[-1])

image_namelist = []

for img in range(len(large_image_stack)):
    
    image = cv2.imread(large_image_stack[img], cv2.IMREAD_COLOR)
    
    name = large_image_stack[img].split('/')[-1].split('.')[0]
    num = large_image_stack[img].split('/')[-1].split('.')[1].split('_')[-1]
    namenum = name + '_' + num
    
    image_namelist.append(large_image_stack[img].split('/')[-1])
    
    patches = patchify(image, (512,512,3), step = 448)  #Step= 448 means 64 overlap
    
  
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):                        
            single_patch_img = patches[i,j,0]     
            cv2.imwrite(cwd + "/data/patches/image_patches/" + namenum + "_" + str(i)+'_'+ str(j)+ ".jpg", single_patch_img)


#print(large_image_stack[0])

#print(len(image_namelist))



for img in range(len(image_namelist)):
    
    
    mask_name = cwd + '/data/original/mask/'+ image_namelist[img].split('/')[-1].split('.jpg')[0]+'_mask.jpg'
    print(mask_name)
    
    image = cv2.imread(mask_name,cv2.IMREAD_COLOR)

    patches = patchify(image, (512,512,3), step = 448)  #Step= 448 means 64 overlap
    
      
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):                        
            single_patch_img = patches[i,j,0]     
            name = image_namelist[img].split('/')[-1].split('.')[0]+'_' + image_namelist[img].split('/')[-1].split('.')[1].split('_')[-1]
            cv2.imwrite(cwd + "/data/patches/mask_patches/" + name +'_' + str(i)+'_'+ str(j)+ ".jpg", single_patch_img)


#print(image_namelist[img].split('/')[-1].split('.')[1].split('_')[-1])

#
print(image_namelist[img].split('/')[-1].split('.')[0]+'_' + image_namelist[img].split('/')[-1].split('.')[1].split('_')[-1])
