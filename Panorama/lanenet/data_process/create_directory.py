#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:00 2022

@author: yolo
"""

import os

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError: 
        print("Error: Failed to create the directory.")
