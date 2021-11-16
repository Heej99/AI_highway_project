#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import Stitching
import GPSInfo 
import PerspectiveTransformer 
import cv2


# gps information extract
gps_file=pd.read_csv("GH011044_Hero6 Black-GPS5.csv")
gpsinfo = GPSInfo.GPSInfo(gps_file)
gps_last_info_df = gpsinfo.match_gps_info()


# make transformed image list
video_path= "GH011044.mp4"
transformer = PerspectiveTransformer.PerspectiveTransformer(video_path=video_path,gps_info=gps_last_info_df)
imageList = transformer.makeImagesList()

# Stitching
stitcher = Stitching.Stitcher(imageList)
panorama = stitcher.Stacking(frame_num=100,start=0)

# save panoram
cv2.imwrite("panorama.jpg",panorama)

