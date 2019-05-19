#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri April 19 18:16:18 2019

@author: gaetanlandreau
"""
import tensorflow as tf
import numpy as np

"""
This file contains the main function used to create the CNN.
"""
def make_batch(X,Y,Z,size_batch,count):
    X_batch=X[count:count+size_batch,:,:,:]
    Y_batch=Y[count:count+size_batch,:,:,:]
    Z_batch=Z[count:count+size_batch,:,:,:]
    return [X_batch,Y_batch,Z_batch]

def create_weights(shape):
    '''
    Return weight variable, regarding the given input shape. Xavier initialization is used. 
    '''
    #return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    Xavier_initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(Xavier_initializer(shape))
def create_bias(size):
    '''
    This function create a bias term
    '''
    return tf.Variable(tf.constant(0.0, shape = [size]))


def convolution(inputs, num_channels, filter_size, num_filters,S):
    'This function implements a convolution layer.'
    
    #Weight and bias term are created from the previously defined function
    weights = create_weights(shape = [filter_size, filter_size, num_channels, num_filters]) 
    bias = create_bias(num_filters)
  
    ## convolutional layer is made
    layer = tf.nn.conv2d(input = inputs, filter = weights, strides= S, padding = 'SAME') + bias
    layer = tf.nn.relu(layer)   #activation function is the ReLu function.
    return layer

def transpose_convolution(inputs, num_channels, filter_size, num_filters,S,factor,depth_factor):
    
    inputs_shape=tf.shape(inputs)   #Size of the inputs layer.
    #Weight and bias term are created for the transposed convolution layer. 
    weights = create_weights(shape = [filter_size, filter_size, num_filters,num_channels]) 
    bias = create_bias(num_filters)
  
    ## convolutional layer
    shape_out=tf.stack([inputs_shape[0],inputs_shape[1]*factor,inputs_shape[2]*factor,inputs_shape[3]//depth_factor])
    layer = tf.nn.conv2d_transpose(inputs, filter = weights,output_shape=shape_out, strides= S, padding = 'SAME')+ bias
    layer = tf.nn.relu(layer)   #activation function is the ReLu function.
    return layer

def convolution_softmax(inputs, num_channels, filter_size, num_filters,S):
    
    
    #Weight and bias term are created for the convolution layer. 
    weights = create_weights(shape = [filter_size, filter_size, num_channels, num_filters]) 
    bias = create_bias(num_filters)
  
    ## convolutional layer
    layer = tf.nn.conv2d(input = inputs, filter = weights, strides= S, padding = 'SAME') + bias
    
    layer = tf.nn.softmax(layer)   #activation function is the softmax function.
    return layer

def batch_normalization(inputs):
    "This function perform a batch normalization."
    layer=tf.layers.batch_normalization(inputs, training=True)
    return layer

def G(x):
    'This function defines the CNN architecture of the function G define in the paper'
    
    #First layer.
    conv1_1 = convolution(x, 1, 3, 64,[1,1,1,1])
    conv1_2 = convolution(conv1_1[:,0:-1,0:-1,:], 64,3,64, [1,2,2,1])
    conv1_out = batch_normalization(conv1_2)
    
    #Second layer.
    conv2_1 = convolution(conv1_out, 64, 3, 128,[1,1,1,1])
    conv2_2=convolution(conv2_1[:,0:-1,0:-1,:],128,3,128,[1,2,2,1])
    conv2_out = batch_normalization(conv2_2)
    
    #Third layer
    conv3_1 = convolution(conv2_out, 128, 3, 256,[1,1,1,1])
    conv3_2=convolution(conv3_1, 256, 3, 256,[1,1,1,1])
    conv3_3=convolution(conv3_2[:,0:-1,0:-1,:],256,3,256,[1,2,2,1])
    conv3_out=batch_normalization(conv3_3)
    
    #Fourth layer
    conv4_1=convolution(conv3_out,256,3,512,[1,1,1,1])
    conv4_2=convolution(conv4_1,512,3,512,[1,1,1,1])
    conv4_3=convolution(conv4_2,512,3,512,[1,1,1,1])
    conv4_out=batch_normalization(conv4_3)
    
    #Fifth layer
    conv5_1=convolution(conv4_out,512,5,512,[1,1,1,1])
    conv5_2=convolution(conv5_1,512,5,512,[1,1,1,1])
    conv5_3=convolution(conv5_2,512,5,512,[1,1,1,1])
    conv5_out=batch_normalization(conv5_3)
    
     #Sixth layer
    conv6_1=convolution(conv5_out,512,5,512,[1,1,1,1])
    conv6_2=convolution(conv6_1,512,5,512,[1,1,1,1])
    conv6_3=convolution(conv6_2,512,5,512,[1,1,1,1])
    conv6_out=batch_normalization(conv6_3)
    
    #Seventh layer
    conv7_1=convolution(conv6_out,512,3,256,[1,1,1,1])
    conv7_2=convolution(conv7_1,256,3,256,[1,1,1,1])
    conv7_3=convolution(conv7_2,256,3,256,[1,1,1,1])
    conv7_out=batch_normalization(conv7_3)
    
    #Eighth layer
    conv8_1=transpose_convolution(conv7_out,256,2,128,[1,2,2,1],2,2)
    conv8_2=convolution(conv8_1,128,3,128,[1,1,1,1])
    conv8_3=convolution(conv8_2,128,3,128,[1,1,1,1])
    
    #A deconvolution layer. 
    conv8_4= transpose_convolution(conv8_3,128,2,128,[1,4,4,1],4,1)
    Z_hat=convolution_softmax(conv8_4,128,1,313,[1,1,1,1])
    Z_hat = tf.clip_by_value(Z_hat,1e-15,100.0)
    return Z_hat
    
def softmax_re_adjusted(z,T):
    
    #f_T = tf.div(tf.exp(tf.log(z)/T),tf.reduce_sum(tf.exp(tf.log(z)/T),axis=-1,keepdims=True))
    f_T = tf.exp(tf.log(z)/T)/tf.reduce_sum(tf.exp(tf.log(z)/T),axis=-1,keepdims=True)
    Z_hat = tf.clip_by_value(f_T,1e-15,100.0)
    return f_T

def H(Z_hat):
    '''This function defines the function H presented in the paper, which allow to pass from 
    class probabilities to point estimates with the mean annealed technique'''
    T=0.38
    f_T_Z=softmax_re_adjusted(Z_hat,T)
    f_T_Z = tf.clip_by_value(f_T_Z,1e-10,100.0)
    
    a=np.arange(313)
    a=np.reshape(a,[1,1,1,313])
  
    b=np.repeat(a,224,axis=1)
    c=np.repeat(b,224,axis=2)

    val=tf.convert_to_tensor(c,dtype=tf.float32)
    
    res=tf.multiply(val,f_T_Z)
    E=tf.reduce_sum(res,axis=-1)
    
    
    return tf.round(E)


def multinomial_cross_entropy_loss(Z_hat,Z,prior):
    
    lambda_value=0.5
    Q=Z_hat.get_shape().as_list()[-1]
    W=Z.shape[2]
    H=Z.shape[1]
    N=Z.shape[0]
    prior=tf.convert_to_tensor(prior,dtype=tf.float32)
    
    term_1=tf.math.scalar_mul(1-lambda_value,prior)
    term_2=lambda_value/Q
    sumt1_t2=tf.add(term_1,term_2)
    w=tf.math.reciprocal(sumt1_t2)
   
    log_Z_hat=tf.log(Z_hat)
    
    prod_term=tf.multiply(log_Z_hat,Z) 
    """Version without the reweighting for inbalance issue"""
    loss=-tf.reduce_sum(prod_term,axis=-1)
   
    return loss