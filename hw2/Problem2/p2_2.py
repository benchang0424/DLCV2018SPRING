import numpy as np
import skimage
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import argparse
import cv2
import scipy.io
from scipy import signal
from scipy import misc

mat = scipy.io.loadmat('filterBank.mat')
filters = np.array(mat['F'])
#print(filters.shape)        49*49*38

image = cv2.imread('mountain.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w, _ = image.shape
print(h)
print(w)
convolutions = np.zeros([38,h*w])

print(convolutions.shape)

for i in range(38):
    convo = signal.convolve2d(gray_image,filters[:,:,i],boundary='symm', mode='same')
    #convolutions.append(convo.reshape(1,-1))
    convolutions[i,:] = convo.reshape(1,-1)

#convolutions = np.array(convolutions).T

convolutions = convolutions.T
kmeans = KMeans(n_clusters=6, max_iter=1000)



lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
lab_image = lab_image.reshape((-1,3))
convolutions = np.concatenate((convolutions, lab_image), axis=1)

kmeans.fit_transform(convolutions)
out = kmeans.labels_.reshape(h,w)
plt.imshow(out,cmap='jet')
#plt.show()
plt.savefig('results/p2_2/m_41.jpg')


