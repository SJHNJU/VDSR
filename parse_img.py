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
def gen_list(rate=2):
    train_img_list = []
    train_label_list = []
    test_img_list = []
    test_label_list = []

    for dir,_,imgs in os.walk('./srdata/label'):
        random.shuffle(imgs)
        for i in range(0,len(imgs)):
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
def img2npy():
    names = ['train_img_list','train_label_list','test_img_list','test_label_list']
    for name in names:
        os.makedirs('./data/'+name)
        with open('./'+name+'.txt','r') as f:
            imgs = []
            J = []
            imgs = f.readlines()
            BATCH_NUM = len(imgs)//64
            for i in range(BATCH_NUM):
                J = []
                for j in range(0,64):
                    img = imgs[i*64+j].replace('\n','')
                    I = scipy.misc.imread(img)
                    I = I.tolist()
                    J.append(I)
                J = np.array(J,dtype='uint8')
                np.save('./data/'+name+'/batch'+str(i)+'.npy',J)
                if i%2 == 0:
                    print('save '+name+' batch'+str(i))


if __name__ == '__main__':
    A = np.load('./data/test_img_list/batch0.npy')
    B = np.load('./data/test_label_list/batch0.npy')
    print(A.shape)
    print(A.dtype)
    plt.imshow(B[1])
    plt.show()
    