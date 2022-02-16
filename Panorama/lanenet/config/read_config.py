#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:38:55 2022

@author: yolo
"""

import warnings
warnings.filterwarnings('ignore')
import yaml

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

## register the tag handler
yaml.add_constructor('!join', join)

def config_(config_path):
	with open(config_path, 'r') as f:
		CFG = yaml.load(f)
	return CFG

	    
