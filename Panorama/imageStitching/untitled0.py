# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:02:48 2021

@author: KONIDE
"""

import cv2

import numpy as np
import os

path = "C:/OpenCV/road/"

images = []

print(os.walk(path))

for root, directories, files in os.walk(path):

    for file in files:
       
        img_input = cv2.imread(os.path.join(root, file))   
        images.append(img_input)
        

    

stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, pano = stitcher.stitch(images)
print(len(images))
print(pano.shape)
cv2.imshow('result', pano)

cv2.waitKey(0)
cv2.destroyAllWindows()