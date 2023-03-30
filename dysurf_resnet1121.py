Python 3.9.13 (v3.9.13:6de2ca5339, May 17 2022, 11:37:23) 
[Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> #!/usr/bin/env python3
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

num_models=2
batch_size = 128

train_data = np.loadtxt('/content/Y.csv',delimiter=',',dtype = np.float32)
label_data = -100 * np.loadtxt('/content/X.csv',delimiter=',',dtype = np.float32)
train_data = train_data.reshape(999,5,5,1)
label_data = label_data.reshape(999,81)


n_data = len(train_data)
train_ratio = 0.8

x_train = train_data[:int(n_data*train_ratio),:]
y_train = label_data[:int(n_data*train_ratio),:]

x_test = train_data[int(n_data*train_ratio):,:]
y_test = label_data[int(n_data*train_ratio):,:]

x_train = x_train.reshape(799,5,5,1)
x_train = tf.constant(x_train)

x_test = x_test.reshape(200,5,5,1)
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
    
#x_train, x_test = standardization(x_train, x_test)

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

class ResidualBlock(tf.keras.Model):
    def __init__(self,filters,kernel_size=3,strides=1,padding='SAME'):
        super(ResidualBlock, self).__init__()
        self.conv1=ConvBNRelu(filters=filters, kernel_size=kernel_size, padding='SAME')
        self.conv2=keras.layers.Conv2D(filters=filters,kernel_size=kernel_size, strides=strides,
                                       padding=padding, kernel_initializer='glorot_normal')
        self.batchnorm=tf.keras.layers.BatchNormalization()
    def call(self,inputs,training=False):
        layer=self.conv1(inputs)
        layer=self.conv2(layer)
        layer=self.batchnorm(layer)
        layer=keras.layers.add([layer,inputs])
        layer=tf.nn.elu(layer)
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
        self.dense5 = tf.keras.layers.Dense(units = 81, activation = tf.nn.elu, kernel_initializer = 'he_uniform')

        
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

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1_1= ConvBNRelu(filters=10,kernel_size=[3,3],padding='SAME')
        self.conv1_2=ResidualBlock(filters=10,kernel_size=[3,3],padding='SAME')
        #self.pool1=keras.layers.MaxPool2D(padding='SAME')
        self.conv2_1= ConvBNRelu(filters=20,kernel_size=[3,3],padding='SAME')
        self.conv2_2=ResidualBlock(filters=20,kernel_size=[3,3],padding='SAME')
        #self.pool2=keras.layers.MaxPool2D(padding='SAME')
        #self.conv3_1= ConvBNRelu(filters=128,kernel_size=[3,3],padding='SAME')
        #self.conv3_2=ResidualBlock(filters=128,kernel_size=[3,3],padding='SAME')
        #self.pool3=keras.layers.MaxPool2D(padding='SAME')
        self.pool3_flat=keras.layers.Flatten()
        self.dense4=DenseBNRelu(units=81)
        self.drop4=keras.layers.Dropout(rate=0.4)
        self.dense5=keras.layers.Dense(units=81, activation = tf.nn.elu, kernel_initializer='he_uniform')
    def call(self, inputs, training=False):
        net=self.conv1_1(inputs)
        net=self.conv1_2(net)
        #net=self.pool1(net)
        net=self.conv2_1(net)
        net=self.conv2_2(net)
        #net=self.pool2(net)
        #net=self.conv3_1(net)
        #net=self.conv3_2(net)
        #net=self.pool3(net)
        net=self.pool3_flat(net)
        net=self.dense4(net)
        net=self.drop4(net)
        net=self.dense5(net)
        return net        
        
models=[]

model=hjModel()
model.build(input_shape=(None,5,5,1))
model.summary()
models.append(model)

model=ResNet()
model.build(input_shape=(None,5,5,1))
model.summary()
models.append(model)

training_epoch = 5000
lr_init = 0.3
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr_init,
                                           #                   decay_steps = 500,
                                           #                   decay_rate = 0.96,
                                           #                   staircase = True)
optimizer = tf.keras.optimizers.SGD(lr_init)

hyper = 0.005

@tf.function
def loss_fn(model,images,labels):
    
    logits = model(images,training=True)
    loss = tf.sqrt(tf.reduce_mean(tf.keras.losses.MSE(logits, labels)))

    return loss

@tf.function
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss=loss_fn(model,images,labels)
    return tape.gradient(loss, model.trainable_variables)

@tf.function
def train(models,images,labels):
    sum_loss=0
    for m in models:
        grads=grad(m,images,labels)
        optimizer.apply_gradients(zip(grads,m.trainable_variables))
        loss=loss_fn(m,images,labels)
        sum_loss+=loss
    return sum_loss / num_models


            
   
for epoch in range(training_epoch):
    avg_loss=0.
    test_loss = 0.
    train_step=0
    test_step=0
    for images, labels in train_dataset:
        avg_loss += train(models,images,labels)
        print('\r{:6.2f}'.format((train_step)/len(train_dataset)*100),end='')
        
        train_step+=1
    avg_loss=avg_loss / train_step
    
    
    for images, labels in test_dataset:
        
        test_loss += train(models,images,labels)
        test_step+=1
        print('\r{:6.2f}'.format((test_step)/len(test_dataset)*100),end='')
  
    if (epoch + 1) % 100 == 0:
         print('\rEpoch:','{}'.format(epoch+1),'loss=','{:.8f}'.format(avg_loss),\
               'Test Loss : {:.8f}'.format(test_loss))

print('Learning Finished!')                                    
                                                             
            
            
