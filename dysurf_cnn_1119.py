#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 22:38:25 2022

@author: kweonhyuckjin
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

tf.random.set_seed(42)


batch_size = 128

train_data = np.loadtxt('X.csv',delimiter=',',dtype = np.float32)
label_data = np.loadtxt('Y.csv',delimiter=',',dtype = np.float32)
train_data = train_data.reshape(999,9,9,1)
label_data = label_data.reshape(999,25)


n_data = len(train_data)
train_ratio = 0.8

x_train = train_data[:int(n_data*train_ratio),:]
y_train = label_data[:int(n_data*train_ratio),:]

x_test = train_data[int(n_data*train_ratio):,:]
y_test = label_data[int(n_data*train_ratio):,:]

x_train = x_train.reshape(799,9,9,1)
x_train = tf.constant(x_train)

x_test = x_test.reshape(200,9,9,1)
x_test = tf.constant(x_test)

def normalization(d_train, d_test = None):
    min_vars = np.min(d_train, axis = 0)
    max_vars = np.max(d_train, axis = 0)
    
    if d_test is None:
        return (d_train - min_vars)/(max_vars- min_vars)
    else:
        return (d_train - min_vars)/(max_vars- min_vars), (d_test - min_vars)/(max_vars- min_vars)

def standardization(d_train, d_test = []):
    mean_vars = np.mean(d_train, axis = 0)
    std_vars = np.std(d_train, axis = 0)
    
    if d_test is None:
        return (d_train - mean_vars)/std_vars
    else:
        return (d_train - mean_vars)/std_vars, (d_test - mean_vars)/std_vars
    
x_train, x_test = standardization(x_train, x_test)
y_train , y_test = standardization(y_train, y_test)

train_dataset = tf.data.Dataset.from_tensor_slices(\
               (x_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(\
              (x_test, y_test)).batch(batch_size)
    

class ConvBNRelu(tf.keras.Model):
    def __init__(self, filters, kernel_size = 3, strides = 1, padding = 'SAME'):
        super(ConvBNRelu, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides,
                                           padding = padding, kernel_initializer = 'he_uniform')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs, training = False):
        layer = self.conv(inputs)
        layer = self.batchnorm(layer)
        layer = tf.nn.elu(layer)
        return layer
    
class DenseBNRelu(tf.keras.Model):
    def __init__(self, units):
        super(DenseBNRelu, self).__init__()
        self.dense = tf.keras.layers.Dense(units = units, kernel_initializer = 'he_uniform')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs, training = False):
        layer = self.dense(inputs)
        layer = self.batchnorm(layer)
        layer = tf.nn.elu(layer)
        return layer


class hjModel(tf.keras.Model):
    def __init__(self):
        super(hjModel, self).__init__()
        self.conv1 = ConvBNRelu(filters = 10, kernel_size=3, padding= 'SAME')
        #self.pool1 = tf.keras.layers.MaxPool2D(padding = 'VALID')
        self.conv2 = ConvBNRelu(filters = 10, kernel_size=3, padding= 'SAME')
        #self.pool2 = tf.keras.layers.MaxPool2D(padding = 'VALID')
        #self.conv3 = ConvBNRelu(filters = 128, kernel_size=3, padding= 'SAME')
        #self.pool3 = tf.keras.layers.MaxPool2D(padding = 'SAME')
        self.pool_flat = tf.keras.layers.Flatten()
        self.dense4 = DenseBNRelu(units = 100)
        #self.dense4_1 = DenseBNRelu(units = 25)
        self.drop4 = tf.keras.layers.Dropout(0.4)
        self.dense5 = tf.keras.layers.Dense(units = 25 , kernel_initializer = 'he_uniform')

        
    def call(self, inputs, training = False):
        net = self.conv1(inputs)
        #net = self.pool1(net)
        net = self.conv2(net)
        #net = self.pool2(net)
        #net = self.conv3(net)
        #net = self.pool3(net)
        net = self.pool_flat(net)
        net = self.dense4(net)
        #net = self.dense4_1(net)
        net = self.drop4(net)
        net = self.dense5(net)
        return net
        
        
tf.keras.backend.clear_session()
model = hjModel()
model.build(input_shape = (None, 9, 9, 1))
model.summary()
        

epoch = 1000
lr_init = 0.3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr_init,
                                                              decay_steps = 500,
                                                              decay_rate = 0.96,
                                                              staircase = True)
optimizer = tf.keras.optimizers.SGD(lr_schedule)

hyper = 0.005

@tf.function
def loss_fn(hypothesis, labels):
    loss = tf.sqrt(tf.reduce_mean(tf.keras.losses.MSE(labels, hypothesis)))
  #    + hyper * (tf.nn.l2_loss(model.trainable_variables[0])+ 
  #                tf.nn.l2_loss(model.trainable_variables[2])+ 
  #                tf.nn.l2_loss(model.trainable_variables[4])+ 
  #                tf.nn.l2_loss(model.trainable_variables[6]))
    return loss

for step in range(epoch):
    for x, y in train_dataset:
        
        with tf.GradientTape() as tape:
            pred = model(x) 
            loss = loss_fn(pred, y)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.trainable_variables))
        
    if (step + 1) % 100 == 0:
        print('step : {}\tloss_train : {:5.9f}\t\tloss_test : {:5.9f}'.format(step+1,\
                                                                            loss,\
                                                                                loss_fn(model(x_test), y_test)))
            
            
