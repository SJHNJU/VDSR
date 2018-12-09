import scipy.misc
import os, random
import numpy as np 
import matplotlib.pyplot as plt 

def downsample(rate=2):
    SIZE = 256   
    os.makedirs('./srdata/'+str(rate)+'/')
    for dir,_,imgs in os.walk('./srdata/label/'):
        for img in imgs:
            #read image
            I = scipy.misc.imread(dir+img)
            #create image
            J = np.zeros((SIZE,SIZE,3),dtype='uint8')
            for i in range(0,SIZE//rate+1):
                for j in range(0,SIZE//rate):
                    if rate*i+rate <= SIZE and rate*j+rate <= SIZE:                       
                        J[rate*i:rate*i+rate,rate*j:rate*j+rate,:] = I[rate*i,rate*j,:]
            
            scipy.misc.imsave('./srdata/'+str(rate)+'/'+img,J)
                
#only single rate
#generate the list of train_img/label test_img/label for make npy
def get_list(rate=2):
    train_img_list = []
    train_label_list = []
    test_img_list = []
    test_label_list = []

    for dir,_,imgs in os.walk('./srdata/label'):       
        random.shuffle(imgs)
        for i in range(0,1600):
            if i % 10 == 0:
                test_label_list.append(os.path.join('./srdata/label',imgs[i]))
                #change the train data here
                test_img_list.append(os.path.join('./srdata/'+str(rate),imgs[i]))
            else:           
                train_label_list.append(os.path.join('./srdata/label',imgs[i]))  
                #change the train data here
                train_img_list.append(os.path.join('./srdata/'+str(rate),imgs[i]))
    #write into txt,very ugly
    with open('./train_img_list.txt','w') as f:
        for line in train_img_list:
            f.write(line+'\n') 
    with open('./train_label_list.txt','w') as f:
        for line in train_label_list:
            f.write(line+'\n') 
    with open('./test_img_list.txt','w') as f:
        for line in test_img_list:
            f.write(line+'\n') 
    with open('./test_label_list.txt','w') as f:
        for line in test_label_list:
            f.write(line+'\n') 

#change the img on the list to ./data/.npy as batches
def img2npy(img_num):
    names = ['train_img_list','train_label_list','test_img_list','test_label_list']
    for name in names:
        os.makedirs('./data'+str(img_num)+'/'+name.replace('_list',''))
        with open('./'+name+'.txt','r') as f:
            imgs = []
            imgs = f.readlines()
            BATCH_SIZE = 20
            BATCH_NUM = len(imgs)//BATCH_SIZE
            for i in range(BATCH_NUM):
                J = []
                for j in range(0,BATCH_SIZE):
                    img = imgs[i*BATCH_SIZE+j].replace('\n','')
                    I = scipy.misc.imread(img)
                    I = I.tolist()
                    J.append(I)
                J = np.array(J,dtype='float32') 
                J = J / 255        
                np.save('./data'+str(img_num)+'/'+name.replace('_list','')+'/batch'+str(i)+'.npy',J)
                del J
                if i%2 == 0:
                    print('save '+name.replace('_list','')+' batch'+str(i))


if __name__ == '__main__':
    img2npy(1600)
    