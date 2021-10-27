#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from glob import glob
import cv2
import shutil
import os 

path = "C:/intern_data/"

mask_files = glob(path+'debug_mask/*.jpg')
image_files = glob(path+'debug_image/*.jpg')

## mask pick

mask_list = [] # mask name collect

for file in range(len(mask_files)):
  
    mask = cv2.imread(mask_files[file])
    
    if mask.sum() != 0 :
        mask_name = mask_files[file]     
        shutil.copy(mask_name,path+'yes_mask/')
        mask_list.append(mask_name.split('/')[2].split('\\')[1])
        

        ## image pick
for image_name in image_files: # 순서 다를 수 있으므로 안에 이름들 가져오기 
    splited_img_name = image_name.split('/')[2].split('\\')[1]
          
    if splited_img_name in mask_list:
        shutil.copy(image_name,path+'yes_image/')

