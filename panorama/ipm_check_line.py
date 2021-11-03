# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 22:17:41 2021

@author: KONIDE

코드 설명
- 원본 이미지에서 투시변환할 부분을 선으로 그려서 미리 확인해보기

"""

import cv2
import numpy as np
import os

path = "C:/OpenCV/"

img = cv2.imread(path+'image-003.jpeg') # 이미지 파일 불러들이기
print(img.shape)
left0 = (960, 380)
left1 = (0,650)
right0 = (960,380)
right1 = (1920,650)
topHeight= 565
width = img.shape[0]
height = img.shape[1]
horizontalTan = 0

down = (horizontalTan, height-horizontalTan)


left = [left0, left1]
right = [right0, right1]
up0 = (int(width/2), topHeight)
up1 = (width, int(topHeight+horizontalTan*width/2))

red = (0,0,255) # (blue,green,red) 0~255

# 좌표계 : 좌측상단(0,0), x축 오른쪽으로 증가 , y측 아래로 증가
# https://copycoding.tistory.com/145 참고
img = cv2.line(img,left0, left1, red, 2 ) # cv.line(img, pt1, pt2, 기타 옵션)  pt1 : 시작점, pt2 : 종료점
img = cv2.line(img, right0, right1, red, 2)
img = cv2.line(img, up0, up1, red, 2)
#img = cv2.line(img, down, red, 2)

cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()

