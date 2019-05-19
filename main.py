#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri April 10 18:16:02 2019

@author: gaetanlandreau
"""

from matplotlib import pyplot as plt
import argparse
import numpy as np
import tensorflow as tf
import cv2 as cv
import os


""" The following file have been made to contain main function required to run the algorithm"""
from preprocessing_functions import create_training_set, create_testing_set,create_set_OpenCV,quantized
from transformeY2Z import create_Z_from_input_Y
from network_functions import G,multinomial_cross_entropy_loss,H,make_batch
from postprocessing_functions import from_Yhat_to_RGBimage,get_Y_hat_fromH

print('----------- Librairies : Loaded -----------')
if __name__ == '__main__':
    
    tf.reset_default_graph()
    
    #Parse different argument
    parser=argparse.ArgumentParser(description='Some input arguments')
    parser.add_argument('--img_folder_input_training',type=str)
    parser.add_argument('--use_already_created_soft_encoding',type=bool,default=True)
    parser.add_argument('--img_folder_input_testing',type=str,default='./images_test')
    parser.add_argument('--img_folder_output_res',type=str,default='./images_res')
    parser.add_argument('--nb_epoch',type=int,default=10)
    parser.add_argument('--size_batch',type=int,default=1)
    args = parser.parse_args()
    
    
    #Folder containing the image. 
    my_folder_training=args.img_folder_input_training
    
    #Folder containing test image.
    my_folder_testing=args.img_folder_input_testing
   
    #Training data are obtained from our image folder. 
    [X,Y]=create_set_OpenCV(my_folder_training)
    print('----------- Training set : Loaded -----------')
    #Testing data are obtained from our image test folder. 
    [X_test,Y_test]=create_set_OpenCV(my_folder_testing)
    print('----------- Test set : Loaded -----------')

    #The (a,b) channels are quantized (for computing accuracy)
    Y_test_quantized=quantized(Y_test)
    
    #Output folder where the indermediaries images will be stored.
    img_folder_output_res=args.img_folder_output_res
    
    #The number of image contained in the training folder.
    N=Y.shape[0]
    #The number of image contained in the testing folder.
    N_test=Y_test.shape[0]

    #Dimension of the images
    h=Y.shape[1]
    w=Y.shape[2]
                        
    #Boolean value to know if the soft encoding have to be processed or not
    bool_soft_encode=args.use_already_created_soft_encoding
    
    if bool_soft_encode:
        #Read the gammut file containing the bins they found out. 
        gammut=np.load('pts_in_hull.npy')
        #Create for each image in it's soft-encoding code.
        Z=create_Z_from_input_Y(Y,gammut)
        np.save('soft_encoding.npy',Z)
        print('----------- Gammut : Loaded -----------')
        print('----------- Soft-Encoding : Computed -----------')
        
    else:
        gammut=np.load('pts_in_hull.npy')
        Z=np.load('soft_encoding.npy')
        print('----------- Gammut : Loaded -----------')
        print('----------- Soft-Encoding : Loaded -----------')
    
    prior=np.load('prior_probs.npy')
    print('----------- Prior probabilities : Loaded -----------')
    
    Z_test=create_Z_from_input_Y(Y_test,gammut)   
    print('----------- Soft-Encoding Test: Computed ------------')
    
    x = tf.placeholder(tf.float32, (None,224, 224, 1))
    soft_z=tf.placeholder(tf.float32, (None,224, 224, 313))
    
    x_test=tf.placeholder(tf.float32, (None,224, 224, 1))
    z_test=  tf.placeholder(tf.float32, (None,224, 224, 313))
    
    Z_hat = G(x)
    Z_soft=soft_z
    Y_hat_bin=H(z_test)
    
    #loss=multinomial_cross_entropy_loss(Z_hat,Z,prior)
    loss=multinomial_cross_entropy_loss(Z_hat,Z_soft,prior)
    cost=tf.reduce_mean(tf.reduce_sum(loss,axis=[1,2]),axis=0)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.00003).minimize(cost)
        
   
    #Define the number of epoch 
    nb_epoch=args.nb_epoch
    #Define the size of each batch (The number of batch is given by the ratio N/nb_batch)
    size_batch=args.size_batch
    
    #the number of batch in the training set
    nb_batch=int(N/size_batch)
    #counter inside a batch
    counter=0;
    
    #Store the loss during training
    loss_training_store=[]
    index_store_loss=[]
    #Store the loss during testing
    loss_testing_store=[]
    index_testing_loss=[]
    #Store the accuracy during testing
    acc_store=[]
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        for i in range(nb_epoch):
            print('-------------------------------------------------------')
            print('------------- Epoch number : ' + str(i+1)+' ------------------------')
            print('-------------------------------------------------------')
            for counter in range(0,N,size_batch):
                [X_batch,Y_batch,Z_batch]=make_batch(X,Y,Z,size_batch,counter)
            #sess.run(optimizer, feed_dict = {x: X, z:Z})
                sess.run(optimizer, feed_dict = {x: X_batch, soft_z:Z_batch})
            #lossvalue = sess.run(cost, feed_dict = {x:X, z : Z})
                lossvalue = sess.run(cost, feed_dict = {x:X_batch, soft_z : Z_batch})
                
                print('Iteration: ' + str(N*i+(counter+size_batch)))
                print("Training loss (loss over batch): " + str(lossvalue))
                loss_training_store.append(lossvalue)
                index_store_loss.append(N*i+(counter+size_batch))
            #The whole algorithm is test during training to monitor the learning.
            if (i==0 or np.mod(i+1,100)==0):
                
                out_test=sess.run(Z_hat,feed_dict = {x: X_test})
                lossvalue_test = sess.run(cost, feed_dict = {x:X_test, soft_z : Z_test})
                print('----------------------------------------------')
                print("Testing multinomial cross entropy loss : " + str(lossvalue_test))
                print('Max prob value from Z_hat for pixel(145,45) in the first image : ')
                print(np.amax(out_test[0,145,45,:]))
                print('Corresponding bin index for the max prob from Z_hat for pixel(145,45) : ')
                print(np.argmax(out_test[0,145,45,:]))
            
                y_out_bin=sess.run(Y_hat_bin, feed_dict = {z_test: out_test}) #(N_test,224,224) image containg the predicted bins.
                
                y_hat=get_Y_hat_fromH(y_out_bin,gammut) # (N_test,224,224,2) image containing the corresponding (â,b) predicted                                                           values for each pixels
                
                
                print('Initial ground truth (ab) value for pixel(145,45) in the first image: ')
                print(Y_test[0,145,45,:])
                print('Quantize ground truth (ab) value pixel(145,45) in the first image :')
                print(Y_test_quantized[0,145,45,:])
                print(' Predicted (a*b*) output values for pixel (145,45) for the first image:')
                print(y_hat[0,145,45,:])
                             
                y_bool= ( y_hat==Y_test_quantized).astype(int)
                y_f=np.sum(y_bool,axis=-1)
                nb_correctly_classified_bin=np.count_nonzero(y_f)
                
                #The image is converted into an RGB image and store
                y_hat_out=from_Yhat_to_RGBimage(y_hat,X_test,gammut,i,img_folder_output_res)
               
                print('Number of correcly classified pixels :')
                print(nb_correctly_classified_bin)
                acc=np.float32((nb_correctly_classified_bin/(1.0*h*w)))
                print('Accuracy : ')
                print(acc)
                
                loss_testing_store.append(lossvalue_test)
                index_testing_loss.append(i+1)
                acc_store.append(acc)
                
        #Some learning curves are plot. 
        print('------------- Training : Done -------------')
        plt.figure(1)
        plt.plot(index_store_loss,loss_training_store,'r',label='training cost')
        plt.xlabel('Iterations')
        plt.ylabel('Training cost')
        plt.title('Evolution of the cost function during training phase')
        plt.legend()
        plt.show()
        plt.figure(2)
        plt.plot(index_testing_loss,loss_testing_store,'g',label='testing cost')
        plt.xlabel('Epoch')
        plt.ylabel('Testing cost')
        plt.title('Evolution of the cost function during testing phase')
        plt.legend()
        plt.show()
        plt.figure(3)
        plt.plot(index_testing_loss,acc_store,'b',label='accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy ')
        plt.title('Evolution of the accuracy during testing phase')
        plt.legend()
        plt.show()
            
            
                
                
             
        