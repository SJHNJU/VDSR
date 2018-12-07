import os, argparse
import tensorflow as tf
import numpy as np
import scipy.misc
from model import *
import matplotlib.pyplot as plt


def train(p):
    X = tf.placeholder(dtype=tf.float32, shape=[None,256,256,3])
    Y = tf.placeholder(dtype=tf.float32, shape=[None,256,256,3])

    net = VDSR(X,Y,p)
    #learning rate decrease every 20 epoches by 0.1
    GLOBAL_STEP = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(p.START_LR, GLOBAL_STEP, p.DECAY_STEPS, p.DECAY_RATE, True)

    optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt = optimizer.minimize(net.loss, global_step=GLOBAL_STEP)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, p.MAX_EPOCH):
            for step in range(0,p.BATCH_NUM):
                print("Start training epoch %d batch %d"%(epoch, step))
                img = np.load('./data/train_img/batch'+str(step)+'.npy')
                label = np.load('./data/train_label/batch'+str(step)+'.npy')
                feed_dict = {X: img, Y: label}
                _,l,lr = sess.run([opt, net.loss, learning_rate], feed_dict=feed_dict)
                print("[epoch %d batch %d] loss %.4f learning_rate %.6f"%(epoch, step, np.sum(l)/64), lr)
                del img,label,
                if step == p.BATCH_NUM-1:
                    output = sess.run(net.output,feed_dict=feed_dict)
                    np.save('./output/epoch'+str(epoch)+'.npy',output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth',default=20)
    parser.add_argument('--BATCH_NUM',default=30)
    parser.add_argument('--START_LR',default=0.1)
    parser.add_argument('--DECAY_STEPS',default=30*20)  #learning rate decrease every 20 epochs
    parser.add_argument('--DECAY_RATE',default=0.1)
    parser.add_argument('--MAX_EPOCH',default=120)
    args = parser.parse_args()

    train(args)

    








'''
    root1 = './data/train_img/'
    root2 = './data/train_label/'
    I = np.load(root1+'batch0.npy')
    J = np.load(root2+'batch0.npy')
    plt.imshow(I[12])
    plt.figure()
    plt.imshow(J[12])
    plt.show()
    print(I.shape)
    print(I.dtype)
'''
    
    
    