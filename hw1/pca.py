import numpy as np
import skimage
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import sys

images_path = sys.argv[1]
#target_path = sys.argv[2]


def plot(data,title,imgname):
    data -= np.min(data)
    data /= np.max(data)
    #data = (data*255).astype(np.uint8).reshape(600,600,3)
    #io.imsave('reconstruction.jpg', data)
    data = (data*255).astype(np.uint8).reshape(56,46)
    fig=plt.figure()
    plt.imshow(data,cmap='gray')
    plt.title(title)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    fig.savefig(imgname)
    #io.imsave(imgname, data,cmap='gray')


def pca(U,img,mean,n):
    img = img - mean
    #print("img shape: ", img.shape)
    out = U[:,:n].T.dot(img.T)
    return out


def reconstruct(U,weight,mean,k):
    img = U[:,:k].dot(weight)
    img = img + mean
    return img


imgs = []
train, test = [],[]
label_0,label_1,label_2 = [],[],[]
train_0,train_1,train_2 = [],[],[]
train_label,test_label = [],[]

for i in range(40):
    for j in range(10):
        #filename = images_path +'/'+ str(i+1) + '.png'
        filename = images_path +'/'+str(i+1) +'_'+ str(j+1) + '.png'
        im = io.imread(filename)
        im = np.array(im)
        im = im.reshape(-1,)
        if(j<6):
            train.append(im)
            train_label.append(i)
            if(j==0 or j==1):
                train_0.append(im)
                label_0.append(i)
            elif(j==2 or j==3):
                train_1.append(im)
                label_1.append(i)
            elif(j==4 or j==5):
                train_2.append(im)
                label_2.append(i)
        else:
            test.append(im)
            test_label.append(i)

train = np.array(train)
test = np.array(test)
test_label = np.array(test_label)
train_label = np.array(train_label)
#imgs = imgs.T
print(train.shape)
print(test.shape)


X_mean = train.mean(axis=0)
print(X_mean.shape)
X = train-X_mean
#X = X.T        #1080000*415
#target_img = train[0:3,:]

#U, s, Vt = np.linalg.svd(X, full_matrices=False)
#weight = pca(U,target_img,X_mean,3)
#print(target_img.shape)
#print(weight.shape)


# problem 2-3
label_0 = np.array(label_0)
label_1 = np.array(label_1)
label_2 = np.array(label_2)
train_0 = np.array(train_0)
train_1 = np.array(train_1)
train_2 = np.array(train_2)


k_near = [1, 3, 5]
dim = [3, 50, 159]

def KNN(data,label,val,val_label,n,k):
    knn = KNeighborsClassifier(n_neighbors=k)
    mean = data.mean(axis=0)
    U, s, Vt = np.linalg.svd(data.T, full_matrices=False)
    reduced_img = pca(U,data,mean,n)
    reduced_val = pca(U,val,mean,n)

    knn.fit(reduced_img.T, label)
    #correct=len(val_label)-np.count_nonzero((knn.predict(reduced_val.T))-val_label)
    correct = knn.score(reduced_val.T,val_label)
    print("k=%d, n=%3d, score=%.5f " %(k,n,correct))
    return correct

data_1 = np.concatenate((train_0,train_1),axis=0)
data_2 = np.concatenate((train_1,train_2),axis=0)
data_3 = np.concatenate((train_0,train_2),axis=0)
y_1 = np.concatenate((label_0,label_1),axis=0)
y_2 = np.concatenate((label_1,label_2),axis=0)
y_3 = np.concatenate((label_0,label_2),axis=0)
print (data_1.shape)
print (y_1.shape)
for k in k_near:
    for n in dim:
        crt = KNN(data_1,y_1,train_2,label_2,n,k)
print()
for k in k_near:
    for n in dim:
        crt = KNN(data_2,y_2,train_0,label_0,n,k)
print()
for k in k_near:
    for n in dim:
        crt = KNN(data_3,y_3,train_1,label_1,n,k)
print()
print("total : ")
KNN(train,train_label,test,test_label,50,1)


"""
# problem 2-1
#print(s)
#plot(X_mean,'mean','mean.png')
#plot(U[:,0],'eig1','eig1.png')
#plot(U[:,1],'eig2','eig2.png')
#plot(U[:,2],'eig3','eig3.png')

# problem 2-2
require = [3,50,100,239]
target_img = train[0,:]
print(target_img)

#fig=plt.figure()
for i,j in enumerate(require):
    weight = pca(U,target_img,X_mean,j)
    re_img = reconstruct(U,weight,X_mean,j)
    mse = np.mean((re_img - target_img)**2)
    #print(mse)
    imgname = "n = " + str(j) +", MSE = "+str(mse)
    name = "n = " + str(j) + ".png"
    data = re_img
    data -= np.min(data)
    data /= np.max(data)
    data = (data*255).astype(np.uint8).reshape(56,46)
    fig=plt.figure()
    #ax  = fig.add_subplot(2, 2, i+1)
    #ax.set_title(imgname)
    plt.imshow(data,cmap='gray')
    plt.title(imgname)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    fig.savefig(name)
    #matplotlib.image.imsave('name.png', array)
    #io.imsave(imgname, data,cmap='gray')

"""

