# -*- coding: utf-8 -*-
"""
Colorization models for deepzipper

Author: Jasmin Omanovic
"""

import tensorflow as tf
from utils import load_and_preprocess_single 
import matplotlib.pyplot as plt
import numpy as np
import random
import time 
import os

# LOAD AND PREPROCESS DATA

image_folder = 'train_images'
image_paths = [os.path.join('train_images', image_name) for image_name in os.listdir(image_folder)]
n_paths = len(image_paths)
batch_size = 32
buffer_size = 1000

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_generator = tf.data.Dataset.from_tensor_slices(image_paths[:int(0.8*n_paths)])
train_generator = train_generator.map(load_and_preprocess_single, num_parallel_calls=AUTOTUNE).shuffle(buffer_size)
train_generator = train_generator.batch(batch_size)
train_generator = train_generator.prefetch(buffer_size=AUTOTUNE)

test_generator = tf.data.Dataset.from_tensor_slices(image_paths[int(0.8*n_paths):])
test_generator = test_generator.map(load_and_preprocess_single, num_parallel_calls=AUTOTUNE).shuffle(buffer_size)
test_generator = test_generator.batch(batch_size)
test_generator = test_generator.prefetch(buffer_size=AUTOTUNE)


# DEFINE MODELS

# BASELINE

def ConvNet():
    model = Sequential()
    model.add(InputLayer(input_shape=(32, 32, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='adam', loss='mse')
    
    return model

class ConvNet_Rec(tf.keras.Model):
    """
    Convolutional Auto-Encoder (Reconstruction):
        Encoder: Conv2D
        Decoder: Conv2D + Depth2Space (Pixel shuffle)
    """
    def __init__(self, input_shape=(32,32,1)):
        super(ConvNet_Rec, self).__init__()
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), activation='relu'),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2, 2), activation='relu')])

        encoder_shape = self.encoder.layers[-1].output_shape
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(encoder_shape[1:])),
            tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=(1, 1), activation='relu', padding='same'),
            tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=(1, 1), activation='relu', padding='same'),
            tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=(1, 1), activation='relu', padding='same'),
            tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1, 1), activation='relu', padding='same'),
            tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2)),
            tf.keras.layers.Conv2D(filters=1, kernel_size=2, strides=(1, 1), activation='relu', padding='same')])  

        

    def call(self, image):
        encoded_image = self.encoder(image)
        final_image = self.decoder(encoded_image)
        return final_image
      

# SET OPTIMIZER
          
optimizer = tf.keras.optimizers.Adam(1e-3)

def compute_loss(model, image):
    pred_image = model(image)
    return tf.keras.losses.mean_absolute_error(pred_image, image)

def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))   
  
    
# SAMPLE FROM MODEL
    
def generate_and_save_images(model, epoch, n_samples=3):
    fig = plt.figure(figsize=(32,32))    
    for i in range(n_samples):
        sample_ix = random.randint(0, len(image_paths))
        image_path = image_paths[sample_ix]
        test_input = load_and_preprocess_single(image_path)
        test_input = np.expand_dims(test_input, 0)
        pred_image = model(test_input)

        plt.subplot(2, 2*n_samples, 2*(i+1))
        plt.imshow(pred_image[0,:,:,0], cmap='gray')
        plt.subplot(2, 2*n_samples, 2*i+1)
        plt.imshow(test_input[0,:,:,0], cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# TRAIN MODEL
    
def train(model, train_generator, test_generator, epochs=5, sample=False):
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for step, train_x in enumerate(train_generator):
            gradients, loss = compute_gradients(model, train_x)
            apply_gradients(optimizer, gradients, model.trainable_variables)
            if step%10 == 0 and sample: generate_and_save_images(model, epoch)
        end_time = time.time()
        if epoch % 1 == 0:
            print('Epoch: {}, time elapse for current epoch {}'.format(epoch, end_time - start_time))
            if sample: generate_and_save_images(model, epoch)
                
    return model

if __name__ == '__main__':
    model = train(ConvNet_Rec(), train_generator, test_generator)
