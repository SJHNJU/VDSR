import os, time, argparse
import tensorflow as tf
import numpy as np
import scipy.misc
from MODEL import *
import matplotlib.pyplot as plt


def train(p):
    BASE_LR = 0.0001
    GLOBAL_STEP = 30
    LR_RATE = 0.1
    LR_STEP_SIZE = 120
    MAX_EPOCH = 120

    X = tf.placeholder(dtype=tf.float32, shape=[None,256,256,3])
    Y = tf.placeholder(dtype=tf.float32, shape=[None,256,256,3])

    net = VDSR(X,Y,p)

    learning_rate = tf.train.exponential_decay(BASE_LR,30,1200,0.1,True)

    optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt = optimizer.minimize(net.loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, MAX_EPOCH):
            for step in range(0,30):
                img = np.load('./data/train_img_list/batch'+str(step)+'.npy')
                label = np.load('./data/train_label_list/batch'+str(step)+'.npy')
                feed_dict = {X: img, Y: label}
                _,l,lr = sess.run([opt, net.loss, learning_rate], feed_dict=feed_dict)
                print("[epoch %d batch %d] loss %.4f"%(epoch, step, np.sum(l)/64))
                del img,label
                if step == 29:
                    output = sess.run(net.output,feed_dict=feed_dict)
                    np.save('./output/epoch'+str(epoch)+'.npy',output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_shape',default=(256,256,3))
    parser.add_argument('--EPOCH',default=120)
    parser.add_argument('--depth',default=20)
    parser.add_argument('--BATCH_SIZE',default=64)
    args = parser.parse_args()

    train(args)
    