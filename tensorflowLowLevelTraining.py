#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:09:53 2018

@author: hsn1997
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
 
## max min scalar on parameters
X_scaler = MinMaxScaler(feature_range=(0,1))
 
## Preprocessing the dataset
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.fit_transform(X_test)
 
## One hot encode Y
onehot_encoder = OneHotEncoder(sparse=False)
Y_train_enc = onehot_encoder.fit_transform(Y_train.reshape(-1,1))
Y_test_enc = onehot_encoder.fit_transform(Y_test.reshape(-1,1))

# Define Model Parameters
learning_rate = 0.01
training_epochs = 1000
 
# define the number of neurons
layer_1_nodes = 150
layer_2_nodes = 150
 
# define the number of inputs
num_inputs = X_train_scaled.shape[1]
num_output = len(np.unique(Y_train, axis = 0)) 
 
# Define the layers
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape= (None, num_inputs))
 
with tf.variable_scope('layer_1'):
    weights = tf.get_variable('weights1', shape=[num_inputs, layer_1_nodes], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias1', shape=[layer_1_nodes], initializer = tf.zeros_initializer())
    layer_1_output =  tf.nn.relu(tf.matmul(X, weights) +  biases) 
 
with tf.variable_scope('layer_2'):
    weights = tf.get_variable('weights2', shape=[layer_1_nodes, layer_2_nodes], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias2', shape=[layer_2_nodes], initializer = tf.zeros_initializer())
    layer_2_output =  tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)
 
with tf.variable_scope('output'):
    weights = tf.get_variable('weights3', shape=[layer_2_nodes, num_output], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias3', shape=[num_output], initializer = tf.zeros_initializer())
    prediction =  tf.matmul(layer_2_output, weights) + biases
 
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape = (None, num_output))#use 1 instead of num output unless one hot encoding??
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = prediction))
 
with tf.variable_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
 
with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(Y, axis =1), tf.argmax(prediction, axis =1) )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# Logging results
with tf.variable_scope("logging"):
    tf.summary.scalar('current_cost', cost)
    tf.summary.scalar('current_accuacy', accuracy)
    summary = tf.summary.merge_all()
    
# Initialize a session so that we can run TensorFlow operations
 
with tf.Session() as session:
 
    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())
 
    # create log file writer to record training progress.
    training_writer = tf.summary.FileWriter(r'C:\data\temp\tf_try\training', session.graph)
    testing_writer = tf.summary.FileWriter(r'C:\data\temp\tf_try\testing', session.graph)
 
    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):
 
        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X:X_train_scaled, Y:Y_train_enc})
 
        # Every 5 training steps, log our progress
        if epoch %5 == 0:
            training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_train_scaled, Y: Y_train_enc})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_test_scaled, Y: Y_test_enc})
 
            #accuracy
            train_accuracy = session.run(accuracy, feed_dict={X: X_train_scaled, Y: Y_train_enc})
            test_accuracy = session.run(accuracy, feed_dict={X: X_test_scaled, Y: Y_test_enc})
 
            print(epoch, training_cost, testing_cost, train_accuracy, test_accuracy )
 
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch) 
 
    # Training is now complete!
    print("Training is complete!\n")
 
    final_train_accuracy = session.run(accuracy, feed_dict={X: X_train_scaled, Y: Y_train_enc})
    final_test_accuracy = session.run(accuracy, feed_dict={X: X_test_scaled, Y: Y_test_enc})
 
    print("Final Training Accuracy: {}".format(final_train_accuracy))
    print("Final Testing Accuracy: {}".format(final_test_accuracy))
 
    training_writer.close()
    testing_writer.close()   
    
    
 