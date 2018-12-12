import scipy.misc
import os, random
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

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
        for i in range(0,640):
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


def save_img():
    A = np.load('./data640/train_img/batch8.npy')
    for i in range(0, 64):
        I = A[i]
        scipy.misc.toimage(I,cmax=1.0,cmin=0.0).save('./train_img/'+str(i)+'.jpg')

    B = np.load('./data640/train_label/batch8.npy')
    for i in range(0, 64):
        I = B[i]
        scipy.misc.toimage(I,cmax=1.0,cmin=0.0).save('./train_label/'+str(i)+'.jpg')


def get_gray():
    names = ['train_img_list','train_label_list']
    for name in names:
        os.makedirs('./gray/'+name.replace('_list',''))
        with open('./'+name+'.txt','r') as f:
            imgs = []
            imgs = f.readlines()
            for i in range(0,len(imgs)):
                img = imgs[i].replace('\n','')
                I = Image.open(img).convert('LA')
                I.save('./gray/'+name.replace('_list','')+'/'+str(i)+'.png')    

def get_gray_npy():
    names = ['train_img_list','train_label_list']
    for name in names:
        os.makedirs('./gray575/'+name.replace('_list',''))
        BATCH_SIZE = 25
        BATCH_NUM = 575//BATCH_SIZE
        for i in range(1,BATCH_NUM+1):
            J = []
            for j in range(1,BATCH_SIZE+1):
                I = scipy.misc.imread('./gray/'+name.replace('_list','')+'/'+str(25*(i-1)+j)+'.png')
                I = I[:,:,0]
                I = I.tolist()
                J.append(I)
            J = np.array(J,dtype='float32') 
         
            np.save('./gray575/'+name.replace('_list','')+'/batch'+str(i)+'.npy',J)
            del J

    

if __name__ == '__main__':
    A  = np.load('./gray575/train_img/batch2.npy')
    B  = np.load('./gray575/train_label/batch2.npy')
    print(A.shape)
    print(A.dtype)
    print(np.max(B))
    print(np.min(B))
    I = A[2]
    I = I/255
    #scipy.misc.toimage(I,cmax=1.0,cmin=0.0).save('./test.jpg')
    plt.imshow(I,cmap='gray')
    plt.show()
   # get_gray()
    