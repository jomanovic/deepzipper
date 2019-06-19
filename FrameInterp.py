# -*- coding: utf-8 -*-
"""
Frame intrpolation models for deepzipper

Author: Jasmin Omanovic
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, UpSampling2D
from utils import load_and_preprocess_seq
import matplotlib.pyplot as plt
import numpy as np
import random
import time 
import os

# LOAD AND PREPROCESS DATA

image_folder = 'train_images'
sorted_names = sorted(os.listdir(image_folder),key=lambda x: x.split('.')[0])
single_paths = [os.path.join('train_images', image_name) for image_name in sorted_names]
seq_paths = [[single_paths[i],single_paths[i+1],single_paths[i+2]] for i in range(len(single_paths)-2)]
n_paths = len(seq_paths)

batch_size = 3*32
buffer_size = 300*32

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_generator = tf.data.Dataset.from_tensor_slices(seq_paths[:int(0.8*n_paths)])
train_generator = train_generator.map(load_and_preprocess_seq, num_parallel_calls=AUTOTUNE).shuffle(buffer_size)
train_generator = train_generator.batch(batch_size)
train_generator = train_generator.prefetch(buffer_size=AUTOTUNE)

test_generator = tf.data.Dataset.from_tensor_slices(seq_paths[int(0.8*n_paths):])
test_generator = test_generator.map(load_and_preprocess_seq, num_parallel_calls=AUTOTUNE).shuffle(buffer_size)
test_generator = test_generator.batch(batch_size)
test_generator = test_generator.prefetch(buffer_size=AUTOTUNE)

# DEFINE MODELS

class LSTMConvNet_v1(tf.keras.Model):
    """
    LSTM Convolutional Auto-Encoder:
        Encoder: LSTM + ConvLSTM2D
        Decoder: UpSampling2D + ConvLSTM2D
    """
    def __init__(self, input_shape=(3, 32, 32, 1)):
        super(LSTMConvNet_v1, self).__init__()
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(3,32,32,1)),
            tf.keras.layers.TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2), padding='same')),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True)),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True)),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True)),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True)),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True))])
            
        encoder_shape = self.encoder.layers[-1].output_shape
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=encoder_shape[1:]),
            tf.keras.layers.TimeDistributed(UpSampling2D((2, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(32, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(UpSampling2D((2, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(32, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(UpSampling2D((2, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(32, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(UpSampling2D((2, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(32, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(UpSampling2D((2, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(16, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(Conv2D(1, (3, 3), strides=(1, 1), padding='same'))])
    
    def call(self, image):
        encoded_image = self.encoder(image)
        final_image = self.decoder(encoded_image)
        return final_image

class LSTMConvNet_v2(tf.keras.Model):
    """
    LSTM Convolutional Auto-Encoder:
        Encoder: LSTM + ConvLSTM2D
        Decoder: Depth2Space (Pixel Shuffle) + ConvLSTM2D
    """  
    def __init__(self, input_shape=(3, 32, 32, 1)):
        super(LSTMConvNet_v2, self).__init__()
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(3,32,32,1)),
            tf.keras.layers.TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2), padding='same')),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True)),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True)),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True)),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True)),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True))])
            
        encoder_shape = self.encoder.layers[-1].output_shape
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=encoder_shape[1:]),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(64, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2))),
            tf.keras.layers.Bidirectional(ConvLSTM2D(32, (3, 3), strides=(1, 1), padding='same', return_sequences=True)),
            tf.keras.layers.TimeDistributed(Conv2D(1, (3, 3), strides=(1, 1), padding='same'))])
        
    def call(self, images):
        encoded_images = self.encoder(images)
        final_image = self.decoder(encoded_images)
        return final_image
    
# SET OPTIMIZER
        
optimizer = tf.keras.optimizers.Adam(1e-3)

def compute_loss(model, images):
    # We use img_t-1, img_t and img_t+1 to predict img_t
    middle_image = images[:,1:2,:,:,:]
    pred_image = model(images)[:,1:2,:,:,:]
    return tf.keras.losses.mean_absolute_error(pred_image, middle_image)

def compute_gradients(model, images):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, images)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))   
    
# SAMPLE MODEL
    
def generate_and_save_images(model, epoch, n_samples=3):
    fig = plt.figure(figsize=(32,32))    
    for i in range(n_samples):
        sample_ix = random.randint(0, len(sorted_names))
        image_paths = seq_paths[sample_ix]
        test_input = load_and_preprocess_seq(image_paths)
        test_input = np.expand_dims(test_input, 0)
        final_image = model(test_input)

        plt.subplot(2, 2*n_samples, 2*(i+1))
        plt.imshow(final_image[0,1,:,:,0], cmap='gray')
        plt.subplot(2, 2*n_samples, 2*i+1)
        plt.imshow(test_input[0,1,:,:,0], cmap='gray')
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
            if step%10 == 0 and sample: 
                if sample: generate_and_save_images(model, epoch)
        end_time = time.time()
        if epoch % 1 == 0:
            print('Epoch: {}, time elapse for current epoch {}'.format(epoch, end_time - start_time))
            if sample: generate_and_save_images(model, epoch)
                
    return model

if __name__ == '__main__':
    print('Training LSTMConvNet_v1...')
    model = train(LSTMConvNet_v1(),train_generator, test_generator)
