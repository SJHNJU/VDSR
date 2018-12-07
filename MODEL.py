import tensorflow as tf
import numpy as np 

'''
important attributes:
    Depth
    Loss function
    SR resolution
    weight initializer
'''
WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()

class VDSR(object):
    def __init__(self, X, Y, p):
        self.X = X
        self.Y = Y
        self.weights = []
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w",[3,3,3,64], initializer=WEIGHT_INITIALIZER)
            bias = tf.get_variable("b",[64],initializer=tf.constant_initializer(0))
            self.weights.append(weights)
            self.weights.append(bias)
            self.output = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.X, weights, strides=[1,1,1,1], padding='SAME'), bias))

        for i in range(p.depth-2):
            scope = "layer%d" % (i+2)
            with tf.variable_scope(scope):
                weights = tf.get_variable("w",[3,3,64,64],initializer=WEIGHT_INITIALIZER)
                bias = tf.get_variable("b",[64],initializer=tf.constant_initializer(0))
                self.weights.append(weights)
                self.weights.append(bias)
                self.output = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.output, weights, strides=[1,1,1,1], padding='SAME'), bias))
        
        with tf.variable_scope("layer"+str(p.depth)):
            weights = tf.get_variable("w",[3,3,64,3],initializer=WEIGHT_INITIALIZER)
            bias = tf.get_variable("b",[3],initializer=tf.constant_initializer(0))
            self.weights.append(weights)
            self.weights.append(bias)
            self.output = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.output, weights, strides=[1,1,1,1], padding='SAME'), bias))

        self.output = self.X + self.output
        #MSE

        self.loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(Y, self.output)))
        for w in self.weights:
            self.loss += tf.nn.l2_loss(w)*1e-4
        
        