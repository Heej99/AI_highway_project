# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:11:10 2021

@author: KONIDE
"""

"""
Created on Tue Aug 24 08:21:42 2021

@author: KONIDE
"""


class Line :
    def __init__(self, data1, data2):
   
        self.line1 = data1
        self.line2 = data2
        #print(self.line1)
    def slope(self):
        (x1, y1), (x2, y2) = self.line1
        (x3, y3), (x4, y4) = self.line2
        
        if (y2-y1) == 0 :
            #print('Ys are equal, m1 = 0')
            m1 = 0
        else:
            m1 = (float(y2)-y1)/(float(x2)-x1)
        
        if (y4-y3) == 0 :
            #print('Ys are equal, m2 = 0')
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
        

class IMP:

  
    def __init__(self, img):
        
        #import cv2
        #img = cv2.imread('c:/OpenCV/image-003.jpeg')     
        self.img = img
        
        #self.topHeight = 565
        #self.height, self.width = 1080, 1920
        
        
        
    def impTransformer(self):  
        
        import numpy as np
        import cv2 
        
        topHeight = 565
        height, width = self.img.shape[:2]
        #print('height', height,'width :', width)
        left = [(960, 380), (0, 650)]
        right = [(960, 380), (1920, 650)]
        up =  [(0, topHeight), (width+1000, topHeight)]
        down =  [(-10000,height), (width+100000, height)]
               
        leftup = Line(left, up)
        leftdown = Line(left, down)
        rightup = Line(right, up)
        rightdown = Line(right, down)
        m1, m2 = leftup.slope()
        b1, b2 = leftup.yintercept(m1,m2)
        p1x, p1y = leftup.findIntersect(m1,m2,b1,b2)
        
        #print('point1 : ', p1x, p1y)
        
       
        
        m1, m2 = leftdown.slope()
        b1, b2 = leftdown.yintercept(m1,m2)
        p2x, p2y = leftdown.findIntersect(m1,m2,b1,b2)
        #print('point2 : ', p2x, p2y)
        
       
        
        m1, m2 = rightup.slope()
        b1, b2 = rightup.yintercept(m1,m2)
        p3x, p3y = rightup.findIntersect(m1,m2,b1,b2)
        #print('point3 : ', p3x, p3y)
        
        m1, m2 = rightdown.slope()
        b1, b2 = rightdown.yintercept(m1,m2)
        p4x, p4y = leftup.findIntersect(m1,m2,b1,b2)
        #print('point4 : ', p4x, p4y)
         
        dst = np.array([[0,0], [0, 565], [1080,0], [1080,565]], dtype=np.float32)
        src = np.array([ [p1x,p1y], [p2x,p2y], [p3x,p3y], [p4x,p4y]], dtype=np.float32)
        mtrx = cv2.getPerspectiveTransform(src, dst)
        
        outimg = cv2.warpPerspective(self.img, mtrx, (1080,560))
        #cv2.imshow('out_image',outimg)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        return outimg

## -----------------Stitching Process --------------------------


    
    
    

