#!/usr/bin/env python
# coding: utf-8

# # 이미지 패치화 하기 


import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from glob import glob
from PIL import Image
import cv2
import os

path = "C:/intern_data/"
large_image_stack = glob(path+'image/Panorama/*.jpg')

print(len(large_image_stack))



## image patchify
for img in range(len(large_image_stack)):
    
    image = cv2.imread(large_image_stack[img],1)
    patches = patchify(image, (512,512,3), step = 448)  #Step= 448 means 64 overlap
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):   
            single_patch_img = patches[i,j,0] # 데이터가 5개면 shape이 10번 찍힘           
            cv2.imwrite(path+"debug_image/" + "image_" + str(img) + "_" + str(i)+str(j)+ ".jpg", single_patch_img)


# mask file index create
large_image_stack_mask = []

for name in large_image_stack:
    word = path+"mask\\" + name[30:-4] + "_mask"+".jpg"
    large_image_stack_mask.append(word)

print(len(large_image_stack_mask))


## mask patchify
for img in range(len(large_image_stack_mask)):    
    image = cv2.imread(large_image_stack_mask[img],1)   
    patches = patchify(image, (512,512,3), step = 448)  #Step= 448 means 64 overlap
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):                        
            single_patch_img = patches[i,j,0]     
            cv2.imwrite(path+"debug_mask/" + "image_" + str(img) + "_" + str(i)+str(j)+ ".jpg", single_patch_img)

