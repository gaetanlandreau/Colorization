#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:17:54 2019

@author: gaetanlandreau
"""

import os
import numpy as np
import cv2 as cv
import io





def lab_translation_CV(I):
    "This function convert the (a,b) channels from [1,156] to [-110,110]"
    return ((I-1.0)/255.0)*220.0-110.0

def quantized(channel):
    "This function return the quantized version of a channel with grid 10.  ( [-110, -100, ...,110])"
    return (np.matrix.round(channel/10)*10).astype(np.int64);


def create_set_OpenCV(my_folder):
    
    #All images are extracted from the folder given in input. 
    images=[files for files in os.listdir(my_folder)]
    N=len(images)
    data_training=np.zeros([N,224,224,3]).astype(np.uint8)     #N training image in the folder. 
    
    for i in range(len(images)):
        img_resize=cv.resize(cv.imread(my_folder + '/'+images[i]),(224,224))
        data_training[i,:,:,:]=img_resize
        #Each image is translated into the CIE-Lab space
        data_training[i,:,:,:]=cv.cvtColor(img_resize, cv.COLOR_RGB2LAB).astype(np.uint8)
    
    img_train=data_training.astype(np.float32)
    X=img_train[:,:,:,0]   #contain Luminance parameter (its our input training data)
    Y=img_train[:,:,:,1:]    #Contain the a (channel 1) and b(channel 2) components. (label datas)
    
    X=X.reshape(N,224,224,1)    #X is reshaped to get a single channel
    Y=Y.reshape(N,224,224,2)  #Y is also reshaped but with 2 channels. 
    Y[:,:,:,0]=lab_translation_CV(Y[:,:,:,0]) #channel a is translated
    Y[:,:,:,1]=lab_translation_CV(Y[:,:,:,1]) #channel b is also translated
    
    return [X,Y]

