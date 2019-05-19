#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:40:26 2019

@author: gaetanlandreau
"""
import numpy as np
import cv2 as cv
import skimage.color as color
import skimage.io as io





def lab_translation_inverse_CV(I):
    "This function translate b* within  [-110;110] to [-128;127]. "
    return ((I+110.0)/220.0)*255.0+1.0

    
def get_Y_hat_fromH(Y_H,gammut):
    '''
    This function takes the output argument from H, which contain a Nx224x224 matrix, 
    where each [N0,i,j] refers to a bin number (between 1 and 313) for image N0 and return a Nx224x224x2 matrix, 
    where the value [N0,i,j,0] contains the a component (resp. the b one on [N0,i,j,1]). 
    '''
    
    w=Y_H.shape[2]
    h=Y_H.shape[1]
    N=Y_H.shape[0]
   
    Y_hat=np.zeros((N,h,w,2))
   
    for n in range(N):
        for i in range(h):
            for j in range(w):
                corresponding_bin_index=np.int_(Y_H[n,i,j])   
                Y_hat[n,i,j,0]=gammut[corresponding_bin_index,0]
                Y_hat[n,i,j,1]=gammut[corresponding_bin_index,1]
    
    return Y_hat.astype(np.int64)

    
def RGB_image_from_Yhat_OpenCV(Yhat,X):
    
    N=Yhat.shape[0]
    w=Yhat.shape[2]
    h=Yhat.shape[1]
    
    
    img_res=np.zeros((N,h,w,3)).astype(np.uint8)
    X=np.squeeze(X,axis=-1).astype(np.uint8)
    for n in range(N):
        
        #The a and b values we get need to be re-translated onto the correct range before getting transform to RGB
        Yhat[n,:,:,0]=lab_translation_inverse_CV(Yhat[n,:,:,0]).astype(np.uint8)
        Yhat[n,:,:,1]=lab_translation_inverse_CV(Yhat[n,:,:,1]).astype(np.uint8)
      
        #Creation of the final image 
        img_final_Lab=np.zeros((h,w,3)).astype(np.uint8)
        
        img_final_Lab[:,:,1:]=Yhat[n,:,:,:];
        img_final_Lab[:,:,0]=X[n,:,:]
      
        #Transformation from Lab to RGB
        img_res[n,:,:,:]=cv.cvtColor(img_final_Lab, cv.COLOR_LAB2RGB)
       
    return (img_res).astype(np.uint8)

def from_Yhat_to_RGBimage(Y_hat,X,gammut,i,output_folder):
    
    #Translate the output result from the H function to the corresponding Yhat (there is no more bins here)
    #Y_hat=get_Y_hat_fromH(Y_bin_res,gammut) #Y_hat size is [N,224,224,2] where N is the number of image in the batch
    
    img_RGB_final=RGB_image_from_Yhat_OpenCV(Y_hat,X)
    #Save the result in a png image. 
    N=img_RGB_final.shape[0]
    for n in range(N):
        img_RGB_to_store=img_RGB_final[n,:]
        img_RGB_to_store=np.squeeze(img_RGB_to_store,axis=None)
        cv.imwrite(output_folder+'/'+'res_epoch_{}.png'.format(i+1), img_RGB_to_store)
    return img_RGB_final


