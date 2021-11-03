# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 21:17:35 2021

@author: KONIDE
"""

import cv2
 
class homographyMat:
    
    def __init__(self):      
      self.point = []
       
    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x,y), 3, (0,255,0), -1)
            cv2.imshow('image', img)
            self.point.append((x,y))     

    

img = cv2.imread('c:/OpenCV/out_image.jpg')
cv2.imshow('image', img)

getPoint = homographyMat()    
cv2.setMouseCallback('image', getPoint.onMouse)

cv2.waitKey(0)
cv2.destroyAllWindows()

for i in getPoint.point:
    print(i)
    
    
    