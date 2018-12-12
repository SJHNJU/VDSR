import os, argparse
import tensorflow as tf
import numpy as np
import scipy.misc
from model import *


def train(p):
    X = tf.placeholder(dtype=tf.float32, shape=[None,256,256,1])
    Y = tf.placeholder(dtype=tf.float32, shape=[None,256,256,1])

    net = VDSR(X,Y,p)
    #learning rate decrease every 5 epoches by 0.1
    GLOBAL_STEP = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(p.START_LR, GLOBAL_STEP, p.DECAY_STEPS, p.DECAY_RATE, True)

    optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt = optimizer.minimize(net.loss, global_step=GLOBAL_STEP)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if p.MODEL_PATH is not None:
            if os.path.exists(p.MODEL_PATH):
                saver.restore(sess,p.MODEL_PATH)
                print("Model Restored")

        for epoch in range(0, p.MAX_EPOCH):
            for step in range(1,p.BATCH_NUM+1):
                print("Start training epoch %d batch %d"%(epoch, step))
                #change dataset here
                img = np.load(p.dataset+'/train_img/batch'+str(step)+'.npy')
                label = np.load(p.dataset+'/train_label/batch'+str(step)+'.npy')
                img = img.reshape([25,256,256,1])
                label = img.reshape([25,256,256,1])
                feed_dict = {X: img, Y: label}
                _,l,lr = sess.run([opt, net.loss, learning_rate], feed_dict=feed_dict)
                info = "[epoch %d batch %d] loss %.8f learning_rate %.6f"%(epoch, step, l, lr)
                print(info)
                with open('./loss_log.txt','a') as f:
                    f.write(info+'\n') 
                del img,label
                #save the output image
                if step == p.BATCH_NUM:
                    output = sess.run(net.output,feed_dict=feed_dict)
                    output = output.reshape([25,256,256])
                    #files in the output need to be deleted
                    os.makedirs('./output/epoch'+str(epoch))
                    np.save('./output/epoch'+str(epoch)+'/epoch'+str(epoch)+'.npy',output)
                    for i in range(0,p.BATCH_SIZE):
                        I = output[i]
                        I = I/255
                        scipy.misc.toimage(I,cmax=1.0,cmin=0.0).save('./output/epoch'+str(epoch)+'/vdsr'+str(i+1)+'.jpg')
            #save the model
            #files in the saved_models need to be deleted
            os.makedirs('./saved_models/epoch'+str(epoch))
            saver.save(sess,'./saved_models/epoch'+str(epoch)+'/epoch'+str(epoch)+'.ckpt')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth',default=20)
    parser.add_argument('--BATCH_NUM',default=23)
    parser.add_argument('--BATCH_SIZE',default=25)
    parser.add_argument('--START_LR',default=0.001)
    parser.add_argument('--DECAY_STEPS',default=23*5)  #learning rate decrease every 5 epochs
    parser.add_argument('--DECAY_RATE',default=0.1)
    parser.add_argument('--MAX_EPOCH',default=120)
    parser.add_argument('--MODEL_PATH',default=None,type=str)
    parser.add_argument('--dataset',default='./gray575')
    args = parser.parse_args()
    
    train(args)
    