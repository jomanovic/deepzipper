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

image_paths = [os.path.join('train_images', image_name) for image_name in os.listdir('train_images')]
n_paths = len(image_paths)
batch_size = 3*32
buffer_size = 300*32

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_generator = tf.data.Dataset.from_tensor_slices(image_paths[:int(0.8*n_paths)])
train_generator = train_generator.map(load_and_preprocess_single, num_parallel_calls=AUTOTUNE).shuffle(buffer_size)
train_generator = train_generator.batch(batch_size)
train_generator = train_generator.prefetch(buffer_size=AUTOTUNE)

test_generator = tf.data.Dataset.from_tensor_slices(image_paths[int(0.8*n_paths):])
test_generator = test_generator.map(load_and_preprocess_single, num_parallel_calls=AUTOTUNE).shuffle(buffer_size)
test_generator = test_generator.batch(batch_size)
test_generator = test_generator.prefetch(buffer_size=AUTOTUNE)


class ConvNet_Res(tf.keras.Model):
    """
    Convolutional Auto-Encoder (Residual):
        Encoder: Conv2D + BatchNorm
        Decoder: Interpolation (Resize) + Conv2D + BatchNorm
    """
    def __init__(self, input_shape=(32, 32, 1)):
        super(ConvNet_Res, self).__init__()
    
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=(2, 2), activation='relu', padding='same')])

        encoder_shape = self.encoder.layers[-1].output_shape
        
        self.interpolate = tf.keras.Sequential([
            tf.keras.layers.InputLayer(encoder_shape[1:]),
            tf.keras.layers.Lambda(lambda x : tf.image.resize(x, (32, 32)))])
        
        interpolate_shape = self.interpolate.layers[-1].output_shape
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(interpolate_shape[1:]),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=(1,1), padding='same')
        ])
        
    def call(self, image):
        encoded_image = self.encoder(image)
        upscaled_image = self.interpolate(encoded_image)
        residual_image = self.decoder(upscaled_image)
        final_image = upscaled_image + residual_image
        return final_image, residual_image, upscaled_image, encoded_image, image
        

# SET OPTIMIZER
        
optimizer = tf.keras.optimizers.Adam(1e-3)

def compute_loss(model, image):
    final_image, residual_image, upscaled_image, encoded_image, original_image = model(image)
    compression_loss = tf.keras.losses.mean_squared_error(original_image, final_image)
    decompression_loss = tf.keras.losses.mean_squared_error(residual_image, original_image - upscaled_image)
    return compression_loss + decompression_loss

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
        final_image, _, _, _, _ = model(test_input)

        plt.subplot(2, 2*n_samples, 2*(i+1))
        plt.imshow(final_image[0,:,:,0], cmap='gray')
        plt.subplot(2, 2*n_samples, 2*i+1)
        plt.imshow(test_input[0,:,:,0], cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# TRAIN MODEL
    
def train(model, train_generator, test_generator, epochs=10, sample=True):
    if sample: generate_and_save_images(model, 0)
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
    model = train(ConvNet_Res(), train_generator, test_generator)