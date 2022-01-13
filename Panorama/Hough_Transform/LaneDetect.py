#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets # UI 라이브러리 함수를 전달하면, 셀렉트 박스나 슬라이더 조작으로 인수를 변경하면서 함수 실행 가능
import IPython.display as display
from ipywidgets import Layout, Button, Box, Layout, Image, IntSlider, AppLayout

class L_Detect:
    """
    원본에서 원근변환 할 point(left,right)  찾는 class
    """
    
    def __init__(self,img):
        self.img = img
        
    # Edge Detection
    def canny(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #blur = cv2.medianBlur(gray,5)
        canny = cv2.Canny(gray,50,200,3)
        return canny
    
    # find roi
    def region_of_interest(self,img):
        height = img.shape[0]
        triangle = np.array([[(0,690),(1920,630),(960,355)]]) # roi <- 변경가능
        mask = np.zeros_like(img)
        cv2.fillPoly(mask,triangle,255)
        masked_image = cv2.bitwise_and(img,mask)
        return masked_image
    
    
    # Hough transform
    def Hough_transform(self):
        canny_image = self.canny(self.img)
        cropped_image = self.region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image,1,np.pi/180,100,np.array([]),minLineLength=10,maxLineGap=300)
        
        
        # find edge line idx
        min_idx_candidate=lines.argmin(axis=0)[0][0::2]
        min_idx = 0 
        if lines[min_idx_candidate[0]][0][0] <= lines[min_idx_candidate[1]][0][0]:
            min_idx= min_idx_candidate[0]
        else:
            min_idx = min_idx_candidate[1]

        max_idx_candidate=lines.argmax(axis=0)[0][0::2]
        max_idx=0

        if lines[max_idx_candidate[0]][0][0] <= lines[max_idx_candidate[1]][0][0]:
            max_idx = max_idx_candidate[0]
        else:
            max_idx = max_idx_candidate[1]
            
        # find edge Line point
        left = [tuple(lines[min_idx][0][0:2]),tuple(lines[min_idx][0][2:])]
        right = [tuple([1920-lines[min_idx][0][0],lines[min_idx][0][1]]),tuple([1920-lines[min_idx][0][2],lines[min_idx][0][3]])]
        
        return left,right

