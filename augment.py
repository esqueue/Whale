# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:15:09 2018

@author: Alex Shi
"""

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import pandas as pd
#import cv2
import math
import os
import random
#from PIL import Image

"""
Helper functions:

1. Load images
2. Load image labels
3. Convert RGB images to grayscale
4. Resize images to 128*128
"""

def get_image_paths():
    folder = './train'
    files = os.listdir(folder)
    files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    return files

X_img_paths = get_image_paths()
#print(X_img_paths)

def load_labels():
    csv_file = './train.csv'
    data_labels = pd.read_csv(csv_file)
    return data_labels

data_labels = load_labels()
#print (data_labels.head())

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

IMAGE_H = 128
IMAGE_W = 128

def tf_resize_images(X_img_file_paths):
    
    X_data = []
    Y_data = []
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 1))
    tf_img = tf.image.resize_images(X, (IMAGE_H, IMAGE_W), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Each image is resized individually as different images may be of different size
        for index, file_path in enumerate(X_img_file_paths):
            img = mpimg.imread(file_path)
            label = data_labels.loc[index]['Id']

            # Convert to grayscale
            if len(img.shape) > 2:
                img = rgb2gray(img)
                
            img = img.reshape(img.shape[0], img.shape[1], 1)    
            resized_img = sess.run(tf_img, feed_dict = {X: img})
            
            X_data.append(resized_img)
            Y_data.append(label)
            
    # Covert to numpy
    X_data = np.array(X_data, dtype = np.float32)
    
    return X_data, Y_data

X_imgs, Y_labels = tf_resize_images(X_img_paths)

#print(X_imgs.shape) # Check X_imgs is of dimension [9850, IMAGE_H = 128, IMAGE_W = 128, 1]
#print(len(Y_labels)) # Check Y_labels is of length [9850]

"""
Rotation between specified angles (+n images)

"""
def rotate_images_n(X_imgs, start_angle, end_angle, n):
    X_rotate = []
        
    iterate_at = (end_angle - start_angle) / (n - 1)
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None, IMAGE_H, IMAGE_W, 1))
    radian = tf.placeholder(tf.float32, shape = (X_imgs.shape[0]))
    tf_img = tf.contrib.image.rotate(X, radian)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
            
        for i in range(n):
            degrees_angle = start_angle + i * iterate_at
            radian_value = degrees_angle * math.pi / 180  # Convert to radians
            radian_arr = [radian_value] * X_imgs.shape[0]
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    Y_rotate = np.tile(Y_labels, n)
    
    return X_rotate, Y_rotate

# Rotate at 30-degree intervals between 30 to 330 degrees
# Generates 11 new images    
n = 11
START_ANGLE = 30
END_ANGLE = 330

rotated_imgs, rotated_labels = rotate_images_n(X_imgs, START_ANGLE, END_ANGLE, n)

#print(rotated_imgs.shape) # Check rotated_imgs is of dimension [9850*n, IMAGE_H = 128, IMAGE_W = 128, 1]
#print(len(rotated_labels)) # Check rotated_labels is of length [9850*n]

# Pick a random image to check that it has the desired rotations and correct labels
iterate_at = (END_ANGLE - START_ANGLE) / (n - 1)
n_unique = len(Y_labels)

i = random.randint(0, n_unique - 1)

matplotlib.rcParams.update({'font.size': 9})

fig, ax = plt.subplots(figsize = (10, 10))
gs = gridspec.GridSpec(3, 4)
gs.update(wspace = 0.30, hspace = 0.0002)

plt.subplot(gs[0])
plt.imshow(X_imgs[i,:,:,0], cmap=cm.gray)   
plt.title('Base Image: {}'.format(Y_labels[i]))

for j in range(n):
    plt.subplot(gs[j + 1])
    plt.imshow(rotated_imgs[i + j * n_unique,:,:,0], cmap=cm.gray)
    plt.title('{0} deg: {1}'.format((j + 1) * iterate_at, rotated_labels[i + j * n_unique]))

fig.suptitle('Plots of {} and its rotations:'.format(data_labels.iloc[i]['Image']))

plt.show()