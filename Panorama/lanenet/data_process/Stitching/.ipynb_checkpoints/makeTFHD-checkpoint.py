#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:44:58 2022

@author: yolo
"""

import sys 
sys.path.append("/home/ict1234/Desktop/panorama/")
from config.read_config import config_

PATH_CFG = config_("/home/ict1234/Desktop/panorama/config/path.yaml")
PANOPARA_CFG = config_("/home/ict1234/Desktop/panorama/config/panorama_para_init.yaml")

# UUser-Defined Function

import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
import pickle


sys.path.extend(PATH_CFG['FUNC_PATH'])
from Stitching.PerspectiveTransformer import IPM
from create_directory import createDirectory

tfhd_list = []
roi_h1,roi_h2 = PANOPARA_CFG['ROI']['HEIGHT']
roi_w1,roi_w2 = PANOPARA_CFG['ROI']['WIDTH']
leftup_x, leftup_y = PANOPARA_CFG['POINT']['LEFT_UP']
leftdown_x, leftdown_y = PANOPARA_CFG['POINT']['LEFT_DOWN']
rightup_x, rightup_y = PANOPARA_CFG['POINT']['RIGHT_UP']
rightdown_x, rightdown_y = PANOPARA_CFG['POINT']['RIGHT_DOWN']



def transformed_FHD_path(fhd_list,npy_list,fhd_num_list,saveTFHD=False):
# MAKE TFHD FOLDER
    createDirectory(PATH_CFG['DATA_PATH']['TFHD_DIR'])
    
    point_list = [ leftup_x, leftup_y, leftdown_x, leftdown_y, rightup_x, rightup_y, rightdown_x, rightdown_y ]
    yellow = ( 0,255,255)
    
    for idx, (f, n, num) in enumerate(zip(fhd_list,npy_list,fhd_num_list)):
        img = cv2.imread(f)
        npy = np.load(n)
        curr_frame = IPM(img,npy)
        final_df, lane_1_exist,lane_4_exist = curr_frame.numberOfLane()
        #print(idx,lane_1_exist,lane_4_exist)
        
        if lane_1_exist:
            result = curr_frame.findIpmParameter_left(final_df) 
            if result==True:
                pass
            else: 
                point_list[0:4]= result
        
        if lane_4_exist:
            result = curr_frame.findIpmParameter_right(final_df) 
            if result==True:
                pass
            else: 
                point_list[4:]= result

        curr_outimg= curr_frame.impTransformer(point_list) # 이미지 정사영 변환
        curr_cropimg = curr_outimg[roi_h1:roi_h2,roi_w1:roi_w2] # 위에 있는 도로만 사용 -> (250, 1080)
        curr_cropimg = cv2.rotate(curr_cropimg, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전 (1080, 250)
        curr_cropimg = cv2.putText(curr_cropimg,"fhd_"+str(num),(30,900),fontScale = 2,
                                   fontFace =  cv2.FONT_HERSHEY_PLAIN,
                                   color = yellow, thickness = 2)
        
        tfhd_list.append(curr_cropimg)
        
        if saveTFHD:
            cv2.imwrite( PATH_CFG['DATA_PATH']['TFHD_DIR']+ '/tfhd_'+ str(num) + '.jpg', curr_cropimg)
	    
    return tfhd_list
	 
	 
def transformed_FHD_object(fhd_list,npy_list,fhd_num_list,saveTFHD=False):
    
	# MAKE TFHD FOLDER
    createDirectory(PATH_CFG['DATA_PATH']['TFHD_DIR'])
    
    point_list = [ leftup_x, leftup_y, leftdown_x, leftdown_y, rightup_x, rightup_y, rightdown_x, rightdown_y ]
    
    for idx, (img, npy, num) in enumerate(zip(fhd_list,npy_list,fhd_num_list)):
        npy = np.array(npy)
        curr_frame = IPM(img,npy)
        final_df, lane_1_exist,lane_4_exist = curr_frame.numberOfLane()
            
        if lane_1_exist:
            result = curr_frame.findIpmParameter_left(final_df) 
            if result==True:
                pass
            else: 
                point_list[0:4]= result
        
        if lane_4_exist:
            result = curr_frame.findIpmParameter_right(final_df) 
            if result==True:
                pass
            else: 
                point_list[4:]= result

        curr_outimg = curr_frame.impTransformer(point_list) # 이미지 정사영 변환
        curr_cropimg = curr_outimg[roi_h1:roi_h2,roi_w1:roi_w2] # 위에 있는 도로만 사용 -> (250, 960)
        curr_cropimg = cv2.rotate(curr_cropimg, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전 (960, 250)
        tfhd_list.append(curr_cropimg)
        
        if saveTFHD:
            cv2.imwrite( PATH_CFG['DATA_PATH']['TFHD_DIR']+ '/tfhd_'+ str(num) + '.jpg', curr_cropimg)
	    
    return tfhd_list
