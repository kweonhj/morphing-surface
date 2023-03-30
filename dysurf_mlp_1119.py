#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 22:36:29 2022

@author: kweonhyuckjin
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

tf.random.set_seed(0)


batch_size = 64

train_data = np.loadtxt('Y.csv',delimiter=',',dtype = np.float32)
label_data = np.loadtxt('X.csv',delimiter=',',dtype = np.float32)
train_data = train_data.reshape(999,25)
label_data = label_data.reshape(999,81)


n_data = len(train_data)
train_ratio = 0.8

x_train = train_data[:int(n_data*train_ratio),:]
y_train = label_data[:int(n_data*train_ratio),:]

x_test = train_data[int(n_data*train_ratio):,:]
y_test = label_data[int(n_data*train_ratio):,:]


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
y_train, y_test = standardization(y_train, y_test)


train_dataset = tf.data.Dataset.from_tensor_slices(\
               (x_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(\
              (x_test, y_test)).batch(batch_size)

def create_model():
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = 25,
                                    activation = tf.nn.elu,
                                    input_shape = (x_train.shape[1],),
                                    kernel_initializer = 'he_uniform'))
    
    model.add(tf.keras.layers.Dense(units = 50,
                                    activation = tf.nn.elu,
                                    kernel_initializer = 'he_uniform'))
                        
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units = 81,
                                    activation = tf.nn.elu,
                                     kernel_initializer = 'he_uniform'))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dense(units = 32,
    #                                activation = 'relu',
    #                                 kernel_initializer = 'he_uniform'))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.elu,
    #                                kernel_initializer='he_uniform'))
    #model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(units = 81, kernel_initializer = 'glorot_normal'))

    return model

tf.keras.backend.clear_session()

model = create_model()
model.build(input_shape = (None,25))
model.summary()  

epoch = 1000
lr_init = 1.5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr_init,
                                                              decay_steps = 500,
                                                              decay_rate = 0.98,
                                                              staircase = True)
optimizer = tf.keras.optimizers.SGD(lr_schedule)

hyper = 0.01

@tf.function
def loss_fn(hypothesis, labels):
    loss = (tf.reduce_mean(tf.keras.losses.MSE(labels, hypothesis)))\
#        + hyper * (tf.nn.l2_loss(model.trainable_variables[0])+ \
#                   tf.nn.l2_loss(model.trainable_variables[2])+ \
#                   tf.nn.l2_loss(model.trainable_variables[4])+ \
#                   tf.nn.l2_loss(model.trainable_variables[6]))
    return loss

for step in range(epoch):
    for x, y in train_dataset:
        
        with tf.GradientTape() as tape:
            pred = model(x) #여기서 모델은 create_model 에서 만든 model
            loss = loss_fn(pred, y)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.trainable_variables))
        
    if (step + 1) % 100 == 0:
        print('step : {}\tloss_train : {:6.9f}\t\tloss_test : {:6.9f}'.format(step+1,\
                                                                            loss,\
                                                                                loss_fn(model(x_test),y_test)))
            
            
print(tf.reduce_mean(y_train-model(x_train)))
