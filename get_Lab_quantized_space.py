#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:35:34 2019

@author: gaetanlandreau
"""
import numpy as np
import cv2 as cv
import tensorflow as tf
import skimage.color as color
import skimage.io as io
import os
from itertools import product
import argparse
from matplotlib import pyplot as plt
'This file allow to create the log probabilities of each bins in the Lab space gamut'

def make_prob_ab_space_v2(my_folder):
    
    #All images are extracted from the folder given in input. 
    folder=[filename for filename in os.listdir(my_folder)]
    N_folder=len(folder)
    N=10000*N_folder
    data_training=np.zeros([N,224,224,3])     #N training image in the folder. 
    for j in range(N_folder):
        img_j=[filename for filename in os.listdir(my_folder+'/'+folder[j]+ '/')]
       
        for i in range(len(img_j)):
           path=my_folder+'/'+folder[j]+ '/' +img_j[i]
           img_resize=cv.resize(cv.imread(path),(224,224))
        #data_training[i,:,:,:]=img_resize
           data_training[i+j*N_folder,:,:,:] = cv.cvtColor(img_resize, cv.COLOR_BGR2LAB)
    #Images in data_training are converted in Lab space before being normalized.    
    
        print('Folder: ',j)
    X=data_training[:,:,:,0]    #contain Luminance parameter (its our input training data)
    Y=data_training[:,:,:,1:]   #Contain the a (channel 1) and b(channel 2) components. (label datas)
    
    X=X.reshape(N,224,224,1)    #X is reshaped to get a single channel
    Y=Y.reshape(N,224,224,2)  #Y is also reshaped but with 2 channels. ``
    
    hist=np.zeros((23,23))
    for i in range(N):
        print(i)
        hist=hist+get_hist_ab(Y[i,:,:,:])
    
    return hist


def quantized_Lab_space():
    "Create a list of discrete tuples. Values from -110 to 110 with grid 10."
    a=np.arange(-110,120,10)
    b=np.arange(-110,120,10)
    list_lab_space=[[(i,j) for j in b] for i in a]
    print(list_lab_space)
    return list_lab_space

def lab_translation_a(I):
    "This function translate a* within [-127;128] to [-110;100] . "
    return ((I+127)/255)*220-110
    
def lab_translation_b(I):
    "This function translate b* within [-128;127] to [-110;100] . "
    return ((I+128)/255)*220-110

def quantized(channel):
    "This function return the quantized version of a channel with grid 10.  ( [-110, -100, ...,110])"
    return np.matrix.round(channel/10)*10;

def get_hist_ab(channel_ab):
    "Compute the 2D histogram in the quantized ab space. Input image is already in Lab space"
    a=channel_ab[:,:,0]
    b=channel_ab[:,:,1]
    hist, xbins, ybins = np.histogram2d(a.ravel(),b.ravel(),[23,23],[[0,256],[0,256]])
    #hist=cv.calcHist(channel_ab,[0,1],None,[23,23],[0,255, 0,255])
   
    return hist
    
if __name__ == '__main__':
 #Parse different argument
 
    parser=argparse.ArgumentParser(description='Some input arguments')
    parser.add_argument('--img_folder_input',type=str)
    #parser.add_argument('img_folder_output',type='str', default='')
    args = parser.parse_args()
    
    
    #Folder containing the image. 
    my_folder=args.img_folder_input
    
    hist=make_prob_ab_space_v2(my_folder)
    total_sum_hist=np.sum(hist)
    Log_prob=np.log(hist/total_sum_hist)
    
    np.save('histT.npy',hist)
    np.save('LogPT.npy',Log_prob)
    
    print(np.count_nonzero(hist))
    plt.matshow(Log_prob)
    plt.colorbar()
    plt.show()