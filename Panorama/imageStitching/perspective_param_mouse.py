# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 21:17:35 2021

@author: KONIDE

코드 설명 
원본 이미지에서 마우스로 찍은 부분의 좌표 알아내기
    - 투시 변환할 부분을 미리 찍어보는 역할을 함.
"""

import cv2
 
class homographyMat:
    
    def __init__(self):      
      self.point = []
       
    def onMouse(self, event, x, y, flags, param): # 마우스 이벤트 처리 콜백 함수 구현
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x,y), 3, (0,255,0), -1) # 좌표에 초록색 동그라미 표시
            cv2.imshow('image', img)
            self.point.append((x,y))     # 마우스 좌표 저장

    

img = cv2.imread('c:/OpenCV/out_image.jpg')
cv2.imshow('image', img)

getPoint = homographyMat()    
cv2.setMouseCallback('image', getPoint.onMouse)

cv2.waitKey(0)
cv2.destroyAllWindows()

for i in getPoint.point:
    print(i)
    
    
    