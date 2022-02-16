#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:05:01 2022

@author: yolo
"""

def sorted_(upper_path,data_type,num_list):
    new_path_list=[]
    for num in num_list:
        new_path = upper_path+str(num)+'.'+data_type
        new_path_list.append(new_path)
        
    return new_path_list     
