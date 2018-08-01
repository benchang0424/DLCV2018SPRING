import numpy as np
import skimage
#from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import argparse
import cv2
"""
jpgfile = Image.open('HW2/problem2/mountain.jpg')

#print (jpgfile.mode, jpgfile.size, jpgfile.format)
tar_img = np.array(jpgfile)

print(tar_img)
print(tar_img.shape)

"""
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", default=10, required = False, type = int,
    help = "# of clusters")
args = vars(ap.parse_args())

#RGB = True
RGB = False
LAB = True
#LAB = False
#mountain = True
mountain = False
zebra = True
#zebra = False

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
kmeans = KMeans(n_clusters=10, max_iter=1000)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)

rgb_image = image.reshape((-1,3))


if(RGB):
    kmeans.fit_transform(rgb_image)

elif(LAB):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_image = lab_image.reshape((-1,3))
    kmeans.fit_transform(lab_image)

if(mountain):
    out = kmeans.labels_.reshape(417,640)
    plt.imshow(out,cmap='jet')
    plt.savefig('results/p2_1/p2_m_labttt.jpg')

elif(zebra):
    out = kmeans.labels_.reshape(331,640)
    plt.imshow(out,cmap='jet')
    plt.savefig('results/p2_1/p2_z_labttt.jpg')


#plt.show()


#colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
#pca = PCA(n_components=2)
#reduced_data_pca = pca.fit_transform(digits.data)