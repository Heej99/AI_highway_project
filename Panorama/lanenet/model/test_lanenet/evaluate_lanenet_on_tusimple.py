#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:08:32 2022

@author: yolo
"""

# path setting
import sys 
sys.path.append("/home/user/lanenet/")
from config.read_config import config_

PATH_CFG = config_("/home/user/lanenet/config/path.yaml")

# Built-in function
import argparse
import glob
import os
import os.path as ops
import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tqdm
import re

# UUser-Defined Function
import sys # 자체 모듈 import를 위한 경로 추가 (리눅스에서)
sys.path.extend(PATH_CFG['FUNC_PATH'])

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger
from create_directory import createDirectory

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_eval')


def eval_lanenet(path_config=PATH_CFG,npysave=False,outputimage_save=False):
    """

    :param src_dir:
    :param weights_path:
    :param save_dir:
    :return:
    """
    
    weights_path = PATH_CFG['DATA_PATH']['WEIGHT_PATH']
    fhd_dir = PATH_CFG['DATA_PATH']['FHD_DIR']
    npy_dir = PATH_CFG['DATA_PATH']['NPY_DIR']
    output_image_dir = PATH_CFG['DATA_PATH']['IMAGE_OUT_DIR']
        
    # make folder 
    createDirectory(npy_dir)
    createDirectory(output_image_dir)
    
    # make parameter
    npy_list = []
    
    # draw lanenet
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet') # lanenet_back_end.inference
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
    
    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        image_list = glob.glob('{:s}/*.jpg'.format(fhd_dir), recursive=True)
        avg_time_cost = []
        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):

            image_origin = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image_origin, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            t_start = time.time()
            
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            avg_time_cost.append(time.time() - t_start)

            postprocess_result,lane_point_line= postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_origin
            )

            if index % 100 == 0:
                LOG.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                avg_time_cost.clear()
            
            if npysave:
                name = re.findall(r'fhd_\d+',image_path)[0]
                np.save(npy_dir + '/{0}'.format(name),lane_point_line)
            
            npy_list.append(lane_point_line)
            
            if outputimage_save:
                name = re.findall(r'fhd_\d+',image_path)[0]
                cv2.imwrite(output_image_dir+"/lanepoint_{0}.jpg".format(name), postprocess_result['source_image'])

    return npy_list
