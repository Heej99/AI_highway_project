# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 08:21:42 2021

@author: KONIDE
"""

import cv2
import numpy as np
#import os

path = "C:/OpenCV/"

img = cv2.imread(path+'image-003.jpeg')
 

class Line:
    def __init__(self, data1, data2):
        self.line1 = data1
        self.line2 = data2
        
    def slope(self):
        (x1, y1), (x2, y2) = self.line1
        (x3, y3), (x4, y4) = self.line2
  
        if (y2-y1) == 0 :
            print('Ys are equal, m1 = 0')
            m1 = 0
        else:
            m1 = (float(y2)-y1)/(float(x2)-x1)
        
        if (y4-y3) == 0 :
            print('Ys are equal, m2 = 0')
            m2 = 0
        else:
            m2 = (float(y4)-y3)/(float(x4)-x3)
            
        return m1, m2
                    
    def yintercept(self, m1, m2):
        (x1, y1), (x2, y2) = self.line1
        (x3, y3), (x4, y4) = self.line2
        
        if m1 != 0 :
            b1 = y1 - m1*x1
        else :
            b1 = y1
            
        if m2 != 0 :
            b2 = y4 - m2*x4
            
        else: b2 = y4
        
        return b1, b2
    
    def findIntersect(self, m1,m2, b1, b2):
        
        if m1 != 0 | m2 != 0 :
            px = (b2-b1) / (m1-m2)
            py = (b2*m1 - b1*m2)/(m1-m2)
        elif m1 == 0 :
            px = (b1-b2)/m2
            py = b1
        elif m2 == 0 : 
            px = (b2-b1)/m1
            py = b2 
        else :  print('No points')
        
        return px, py
        


topHeight = 565
height, width = img.shape[:2]
left = [(960, 380), (0, 650)]
right = [(960, 380), (1920, 650)]
up =  [(0, topHeight), (width+1000, topHeight)]
down =  [(-10000,height), (width+100000, height)]


leftup = Line(left, up)
m1, m2 = leftup.slope()
b1, b2 = leftup.yintercept(m1,m2)
p1x, p1y = leftup.findIntersect(m1,m2,b1,b2)
print('point1 : ', p1x, p1y)

leftdown = Line(left, down)
m1, m2 = leftdown.slope()
b1, b2 = leftdown.yintercept(m1,m2)
p2x, p2y = leftdown.findIntersect(m1,m2,b1,b2)
print('point2 : ', p2x, p2y)


rightup = Line(right, up)
m1, m2 = rightup.slope()
b1, b2 = rightup.yintercept(m1,m2)
p3x, p3y = rightup.findIntersect(m1,m2,b1,b2)
print('point3 : ', p3x, p3y)

rightdown = Line(right, down)
m1, m2 = rightdown.slope()
b1, b2 = rightdown.yintercept(m1,m2)
p4x, p4y = leftup.findIntersect(m1,m2,b1,b2)
print('point4 : ', p4x, p4y)
 

yellow = (0,255,255)
blue = (255,0,0)
red = (255, 0, 255)

font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fontScale = 1
tickness = 2

img = cv2.line(img, left[0], left[1], red, 2)
img = cv2.line(img, right[0], right[1], red, 2)
img = cv2.line(img, up[0], up[1], red, 2)
img = cv2.line(img, down[0], down[1], red, 2)
img = cv2.putText(img, 'Point1', (int(p1x), int(p1y)), font, fontScale, yellow, tickness, cv2.LINE_AA)
img = cv2.putText(img, 'Point2', (0, height-200), font, fontScale, blue, tickness, cv2.LINE_AA)
img = cv2.putText(img, 'Point3', (int(p3x), int(p3y)), font, fontScale, blue, tickness, cv2.LINE_AA)
img = cv2.putText(img, 'Point4', (int(width/2), int(height/2)), font, fontScale, blue, tickness, cv2.LINE_AA)

dst = np.array([[0,0], [0, 565], [1080,0], [1080,565]], dtype=np.float32)
src = np.array([ [p1x,p1y], [p2x,p2y], [p3x,p3y], [p4x,p4y]], dtype=np.float32)
mtrx = cv2.getPerspectiveTransform(src, dst)


cv2.imshow('input_image', img)
cv2.waitKey()
cv2.destroyAllWindows()

outimg = cv2.warpPerspective(img, mtrx, (1080,560))
cv2.imwrite('c:/OpenCV/out_image.jpg', outimg)
cv2.imshow('c:/OpenCV/out_image.jpg',outimg)
cv2.waitKey()
cv2.destroyAllWindows()

