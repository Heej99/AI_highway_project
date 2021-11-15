# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 16:10:52 2021

@author: KONIDE

코드 설명  
- 이미지를 파노라마 형식으로 붙이도록 실행함.
"""
import Stitching
import cv2


    


stitcher = Stitching.Stitcher()

imageList = stitcher.makeImagesList() # 동영상에서 정사영된 정지 프레임 리스트 추출

images1 = [imageList[4],imageList[6]] # 이어 붙일 처음 이미지 2개 선별

result1 = stitcher.stitch(images1) # 처음 이미지 두개 붙이기 

images2 = [result1,imageList[8]] # 위에서 이어붙인 결과와 다음 이미지 리스트 생성

result2= stitcher.stitch(images2) # 이미지 이어 붙이기

print(result2.shape)

cv2.imshow('result',result2)

cv2.waitKey(0)

cv2.destroyAllWindows()