# -*- coding: utf-8 -*-
"""
Utility functions for deepzipper
Author: Jasmin Omanovic
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import matplotlib.pyplot as plt
from skimage.color import rgb2yuv, yuv2rgb, rgb2grey
import numpy as np
import random
import time
import os

def extract_frames(ffmpeg_path, img_path, video_path, n_frames=10):
    """
    Extracts frames from video_path to img_path.
    """
    error = ""
    print('{} -i {} -framerate {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(ffmpeg_path, video_path, n_frames, img_path))
    failed_call = os.system('C:\ffmpeg\bin\ffmpeg -i {} -framerate {} -vsync 0 -qscale:v 2 {}/%d.jpg'.format(video_path, n_frames, img_path))
    if failed_call:
        error = "Error converting file:{}. Exiting.".format(video_path)
    return error

def frames_to_video(ffmpeg_path, img_path, video_path, n_images=1000):
    """
    Saves video in video_path based on model predictions saved in img_path
    """
    error = ""
    print('{} -framerate 30 -i {}/%d.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p {}/movie.mp4'.format(ffmpeg_path, img_path, video_path))
    failed_call = os.system('{} -framerate 30 -i {}/%d.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p {}/movie.mp4'.format(ffmpeg_path, img_path, video_path))
    if failed_call:
        error = "Error converting images:{}. Exiting.".format(img_path)
    return error

def load_image(image_path, size):
    image = tf.io.read_file(image_path)
    return preprocess_image(image, size)

def preprocess_image(image, size):
    image = tf.image.decode_jpeg(image, channels=3) 
    image = tf.image.resize(image, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return rgb2yuv(image)

def rgb2yuv(image):
    """
    RGB: [0, 255] -> YUV: [-1, 1] 
    """
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255.0) # RGB: [0,255] -> [0,1]
    image = tf.image.rgb_to_yuv(image) # RGB: [0, 1] -> Y: [0,1], U: [-0.5, 0.5], V: [-0.5, 0.5] 
    image = tf.subtract(image, [0.5,0,0]) # Y: [0,1] -> [-0.5, 0.5] 
    return image
    
def yuv2rgb(image):
    """
    YUV: [-1, 1] -> RGB: [0, 255]
    """
    image = tf.add(image, [0.5,0,0])
    image = tf.image.yuv_to_rgb(image)
    return image
    
def visualize_yuv(image):
    """
    Visualize YUV image in RGB scale
    """
    return plt.imshow(tf.clip_by_value(yuv2rgb(image),0,1))

def load_and_preprocess_single(image_path, size=(32,32)):
    image = load_image(image_path, size)
    return image[...,0:1]

def load_and_preprocess_seq(image_list, seq_length=3):
    image_seq = []
    for i in range(seq_length):
        image_seq.append(load_and_preprocess_single(image_list[i]))
    image_seq = tf.stack(image_seq)
    return image_seq