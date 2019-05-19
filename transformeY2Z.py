#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:12:52 2019

@author: gaetanlandreau
"""
import numpy as np


def soft_encoding(Y,gammut):
    """
    This function takes as input an input image Y, size 224x224x2 (ab channel) and the gammut
    (computed by the researchers) and compute the 5 closest bins for each pixels, compute a distance, 
    and return a softmax distribution (soft encoding) for each pixel with 5 values.
    """
    
    a=Y[:,:,0]    #Correspond to the channel a - ground truth
    b=Y[:,:,1]    #Correspond to the channel b - ground truth
    w=a.shape[-1]
    h=a.shape[1]
    Q=313
    
    
    #Matrix of size 224x(2*224) where first column is a channel for the first column 
    #of channel a in Y,and the next column is the b values for the same column of b
    cross_mat=np.empty((h,2*w), dtype=a.dtype)    
   
    #Fill the even column with a, and the odd ones with b. Matrix size is twice the initial one along axis=1
    cross_mat[:,0::2] = a
    cross_mat[:,1::2] = b
    
    #Cross_mat is extended along a fourth dimension, Q=313. Y_extended shape is (224,448,313)
    Y_extended=np.repeat(cross_mat[:, :,np.newaxis], Q, axis=-1)   
    
    #The gammut is resize to a matrix (1,2,Q)
    gammut_inter=np.empty((1,2,Q))
    gammut_inter[0,0,:]=gammut[:,0] #channel a
    gammut_inter[0,1,:]=gammut[:,1] #channel b
   
    #The gammut is resize to a matrix (224,2,313)
    Gammut_extended=np.repeat(gammut_inter,h,axis=0)
    #The gammut is resize to a matrix (224,448,313)
    Gammut_extended=np.tile(Gammut_extended,(1,w,1))
    
    
    Z=np.empty((w,h,5))
    #Final soft encoded matrix
    Z_b=np.zeros((w,h,Q))
    diff=Y_extended-Gammut_extended    #224,448,313 matrix, 
        
    diff_even=diff[:,0::2,:]
    diff_odd=diff[:,1::2,:]
        
    
    norm_res=(diff_even*diff_even)+(diff_odd*diff_odd)#(224,224,313) matrix, which contains [(a-a*)^2+(b-b*)^2]
   
    min_5_index_values = np.argpartition(norm_res,5, axis = -1)[:,:,:5]
    
    min_5_values=np.sort(norm_res,kind='mergesort',axis=-1)[:,:,:5]
   
    min_value_right_order_new=np.empty((w,h,5))
    for i in range(h):
        for j in range(w):
                min_value_right_order_new[i,j,:]=norm_res[i,j,min_5_index_values[i,j,:]]
                
    
    Z=np.apply_along_axis(gauss_kernel1D,-1,min_value_right_order_new) #only on the five smallest distance! 
   
    for i in range(h):
        for j in range(w):
            Z_b[i,j,min_5_index_values[i,j,:]]=Z[i,j,:]
            
    
    return Z_b  
   
def create_Z_from_input_Y(Y, gammut):
    """Some variable are created"""
    N=Y.shape[0]    #The first dimension on Y contains the number of image we have in the batch.
    w=Y.shape[2]   #width of an image  in Y 
    h=Y.shape[1]    #height of an image in Y
    Q=313
    Z=np.empty((N,w,h,Q))   #This matrix will contains for each image the corresponding softencoding, 
                            #used to compute the loss function. 
    """For loop over all image in a batch"""
    for i in range(N):
        
        Z[i,:,:,:]=soft_encoding(Y[i,:,:,:],gammut)  #The ith soft encoding is stored in Z. 
        
    """The matrix Z is returned once we have computed the soft encoding for all images"""
    return Z

        
def gauss_kernel1D(x):
    
    eps=10**-8
    x_sort=sorted(x)
    mean=x_sort[0]
    sigma=0.8*x_sort[1]+eps
    #sigma=eps*eps
    Mean=np.repeat(mean,(x.shape[0],))
    Sigma=np.repeat(sigma,(x.shape[0],))

    res=(1/(np.sqrt(2*np.pi)*Sigma))*(np.exp((-(x-Mean)**2)/(2*(Sigma)**2)))
    return res/sum(res)
    