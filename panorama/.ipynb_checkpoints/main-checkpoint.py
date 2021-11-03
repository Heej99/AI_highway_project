# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 16:10:52 2021

@author: KONIDE
"""
import Stitching
import cv2


    


stitcher = Stitching.Stitcher()

imageList = stitcher.makeImagesList()

images1 = [imageList[4],imageList[6]]

result1 = stitcher.stitch(images1)

images2 = [result1,imageList[8]]

result2= stitcher.stitch(images2)

print(result2.shape)

cv2.imshow('result',result2)

cv2.waitKey(0)

cv2.destroyAllWindows()