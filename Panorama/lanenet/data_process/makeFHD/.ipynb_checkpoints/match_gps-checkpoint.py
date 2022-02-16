#!/usr/bin/env python3
# -*- coding: utf-8 -*-￣
"""
Created on Thu Jan 13 13:04:00 2022

@author: yolo
"""

import numpy as np
import pandas as pd
import cv2

from create_directory import createDirectory


class gps_match:
    def __init__(self,gps_df,video_path,save_fhd_dir):
        self.gps_df = gps_df
        self.video_path = video_path
        self.save_dir = save_fhd_dir
        self.FHDlist = []
        self.fhd_num_list = []
        
    def CalcFrameSkip(self, speed, fps):
        magicNum = 3.6
        skipVal = (magicNum * fps) / speed

        if speed >= 3.6:
            skipVal = (magicNum * fps) / speed
        else:
            skipVal = (magicNum * fps) / 3.6
        return skipVal


    def makeFHDList(self,SkipSec=0,frame_num=500,saveFHD=False):       
                
        cap = cv2.VideoCapture(self.video_path)  # 비디오 객체 생성
        fps = round(cap.get(cv2.CAP_PROP_FPS))  # get frame numbers per seconds
        if (fps == 0) :
            fps = 60 
        
        # define variable
        cap.set(cv2.CAP_PROP_POS_FRAMES,fps*SkipSec)
        currentFrameNumber = fps*SkipSec
        currentFrameNumberTemp = currentFrameNumber
        skipGPS = 18*SkipSec
        current_frame_count=0 # 만들 프레임 개수 
        
        # 저장할 파일 경로 생성
        createDirectory(self.save_dir)
        
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                currentFrameNumber += 1
                #print(currentFrameNumber)
                
                ## gps info mapping
                speed = self.gps_df['GPS (2D speed) [m/s]'][skipGPS]
                
                ## skipping frame
                skipVal = self.CalcFrameSkip(speed,fps)
                if (currentFrameNumber != 1) & ((currentFrameNumber - skipVal) < currentFrameNumberTemp) :
                    continue

                currentFrameNumberTemp = currentFrameNumber
                skipGPS = skipGPS + round(skipVal / fps * 18) #calculate next speed
            
                ## fhd list 생성
                self.FHDlist.append(frame)
                self.fhd_num_list.append(currentFrameNumber)

                
                ## save image
                if saveFHD:
                    cv2.imwrite(self.save_dir+ '/fhd_' + str(currentFrameNumber) + '.jpg', frame)
                    
                ## FHD making early stopping
                current_frame_count+=1 # calculate current frame count
                if current_frame_count ==frame_num:
                    break
                
            if currentFrameNumber == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break

        return self.FHDlist,self.fhd_num_list
