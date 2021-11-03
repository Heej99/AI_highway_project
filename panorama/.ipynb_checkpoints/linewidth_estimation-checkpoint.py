# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:26:29 2021

@author: KONIDE
"""


import cv2
import numpy as np
#import os

path1 = "C:/OpenCV/"
path2 = "C:/OpenCV/line_img/"

#img = cv2.imread(path+'0001202108170118_F.jpg')
img = cv2.imread(path1+'15_S.jpg')

#img2 = img.copy()

print(img.shape)

# 그레이 스케일로 변환
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow('imggray',img)



# 바이너리 이미지로 반들어 검은색 배경에 흰색 전경으로 변환
ret, imthres = cv2.threshold(imggray, 127, 255, cv2.THRESH_BINARY_INV)



'''
# 컨투어에 대한 꼭짓점 좌표만 변환
img2, contour, hierachy = cv2.findContours(imthres, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 컨투어에 대한 꼭짓점 좌표만 변환
img2, contour2, hierachy = cv2.findContours(imthres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# 각 컨투어 개수 출력
print('number of contour : %d(%d) ' % (len(contour), len(contour2)))

# draw every pionts of contour, green
cv2.drawContours(img, contour, -1, (0,255,0), 4)
cv2.drawContours(img2, contour2, -1, (0,255,0), 4)


for i in contour:
    for j in i:
        cv2.circle(img, tuple(j[0]), 1, (255,0,0), -1)
        
for i in contour2:
    for j in i:
        cv2.circle(img2, tuple(j[0]), 1, (255,0,0), -1)
        
# show the results
##cv2.imshow('CHAIN_APPROX_NONE',img)
##cv2.imshow('CHARIN_APPROX_SIMPLE', img2)


canny = cv2.Canny(img, 150,300)

laplacian = cv2.Laplacian(img, -2)
merged = np.vstack((img,laplacian))

scharrx = cv2.Scharr(img, -1,1,0, scale = 10, delta= 50)  # src, depth, dx, dy, [dst, scale, delta, borderType] 
scharry = cv2.Scharr(img, -1,0,1, scale = 10, delta= 50)

merged_schr = np.vstack((scharrx, scharry, laplacian))
   
print(canny.shape)                      
print(laplacian.shape)
cv2.imshow('Scharr', merged_schr)
cv2.imshow('Canny', canny)
#cv2.imshow('Laplacian',merged)
'''
cv2.waitKey()
cv2.destroyAllWindows()
 