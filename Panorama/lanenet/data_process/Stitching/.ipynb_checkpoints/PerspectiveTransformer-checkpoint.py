#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:17:00 2022

@author: yolo
"""
import numpy as np
import cv2
import pandas as pd

import sys 
sys.path.append("/home/ict1234/Desktop/panorama")
from config.read_config import config_

PATH_CFG = config_("/home/ict1234/Desktop/panorama/config/path.yaml")
PANOPARA_CFG = config_("/home/ict1234/Desktop/panorama/config/panorama_para_init.yaml")


class Line :
    """
    투시변환할 부분을 찾아주는 class
    """
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
        #print('기울기:',m1,m2)    
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
        #print('y절편:',b1,b2)
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
        #print('교점:',px,py)
        return px, py
    
class IPM:

  
    def __init__(self, img, npy,para_cfg = PANOPARA_CFG):
        
        self.img = img
        self.npy = npy
        self.lane_1_num = para_cfg['LANE']['LANE_1']
        self.lane_2_num = para_cfg['LANE']['LANE_2']
        self.lane_3_num = para_cfg['LANE']['LANE_3']
        self.lane_4_num = para_cfg['LANE']['LANE_4']
        self.top_height = para_cfg['PERSPECTIVE']['TOPHEIGHT']
        self.start_y = para_cfg['PERSPECTIVE']['START_Y']
        
        #self.topHeight = 565
        #self.height, self.width = 1080, 1920
    
    # 차선 영역을 정해주는 함수
    def distinguish_lane(self,num):
        if num>=0 and num <=self.lane_1_num:
            return 'lane_1'
        elif num>self.lane_1_num and num<=self.lane_2_num:
            return 'lane_2'
        elif num>self.lane_2_num and num <=self.lane_3_num:
            return 'lane_3'
        elif num>self.lane_3_num and num<= self.lane_4_num:
            return 'lane_4'
        else:
            return 'lane_5'
        
    # 영역에 따라 차선을 할당해주는 함라
    def check_lane(self, df):  
        for lane in df['lane'].unique():
            median_x_point = int(df[df['lane']==lane]['x'].median())
            lane_num = self.distinguish_lane(median_x_point)
            df['lane']=df['lane'].apply(lambda x: lane_num if x==lane else x)

        return df
               
    def numberOfLane(self):
        
        lane_1_exist = False
        lane_4_exist = False
        
        if self.npy.size == 0:
            return None, lane_1_exist,lane_4_exist 
        
        else:
            col_names = ['lane','x','y']
            df = pd.DataFrame(self.npy, columns=col_names)
            # 최종 df 생성
            final_df=self.check_lane(df)
            final_df=final_df.sort_values(by=['lane','y'])
            
            if 'lane_1' in final_df['lane'].unique():
                lane_1_exist = True
            if 'lane_4' in final_df['lane'].unique():
                lane_4_exist = True

        return final_df, lane_1_exist,lane_4_exist
        
    def findIpmParameter_left(self,final_df):
        stop = False
        
        # 1차선 정사영 파라미터 
        if len(final_df[(final_df['lane'] == 'lane_1') &(final_df['y'] >= self.start_y)]) < 5:
            stop = True
            return stop
        
        lane,leftup_x,leftup_y = final_df[(final_df['lane'] == 'lane_1') &(final_df['y'] >= self.start_y)].iloc[0]
        lane,leftdown_x,leftdown_y = final_df[(final_df['lane'] == 'lane_1') &(final_df['y'] >= self.start_y)].iloc[4]
        
        return leftup_x,leftup_y,leftdown_x,leftdown_y
    
    def findIpmParameter_right(self,final_df):
        
        stop = False
        
        # 1차선 정사영 파라미터 
        if len(final_df[(final_df['lane'] == 'lane_4') &(final_df['y'] >= self.start_y)]) < 5:
            stop = True
            return stop
        
        # 4차선 정사영 파라미터        
        lane,rightup_x,rightup_y = final_df[(final_df['lane'] == 'lane_4') &(final_df['y'] >= self.start_y)].iloc[0]
        lane,rightdown_x,rightdown_y = final_df[(final_df['lane'] == 'lane_4') &(final_df['y'] >= self.start_y)].iloc[4]
        
        return rightup_x, rightup_y, rightdown_x, rightdown_y 
    
    def impTransformer(self,point_list):  

        # 전체 파라미터
        leftup_x, leftup_y, leftdown_x, leftdown_y, rightup_x, rightup_y, rightdown_x, rightdown_y = point_list
        
        topHeight = self.top_height
        img = self.img
        height, width = img.shape[:2]
        left = [(leftup_x, leftup_y), (leftdown_x, leftdown_y)]
        right = [(rightup_x, rightup_y), (rightdown_x, rightdown_y)]
        up =  [(0, topHeight), (width, topHeight)]
        down =  [(-10000,height), (width+10000, height)]


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

        # 왼쪽위, 왼쪽아래, 오른쪽위, 오른쪽아래
        dst = np.array([[0,0], [0, height], [width,0], [width,height]], dtype=np.float32)
        src = np.array([ [p1x-15,p1y], [p2x-15,p2y], [p3x+20,p3y], [p4x+20,p4y]], dtype=np.float32) 
        mtrx = cv2.getPerspectiveTransform(src, dst)

        # C++ 코드에서 원본 사이즈로 정사영 변환
        transformedFHD = cv2.warpPerspective(img, mtrx, (width,height)) # 투시 변환 -> 네 점에 대해서 정면에서 바라본 위치?

        outimg = cv2.resize(transformedFHD, (960,540))
        #cv2.imshow('out_image',outimg)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        return outimg
