import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

category_list = ['Coast', 'Forest', 'Highway', 'Mountain', 'Suburb']
img_name_list = []

for category in category_list:
	img_name_list += ['train-10/'+category+'/'+img_name for img_name in os.listdir('train-10/'+category)]
print(len(img_name_list))

key_point_dict = []
for img_name in img_name_list:
	img = cv2.imread(img_name)
	surf = cv2.xfeatures2d.SURF_create()
	kp, des = surf.detectAndCompute(img, None)
	#print(len(kp), des.shape)
	for des_i in des:
		key_point_dict.append(des_i)

interst_points = np.array(key_point_dict)
print(interst_points.shape)

kmeans = KMeans(n_clusters=50, max_iter=5000).fit(interst_points)
# np.save('centroids.npy', kmeans.cluster_centers_)

labels = kmeans.labels_
print(labels.shape)

pca = PCA(n_components=3)
pca_points = pca.fit_transform(interst_points)
x_centers = pca.transform(kmeans.cluster_centers_[:6])
y_centers = np.arange(6)
print(pca_points.shape)
print(x_centers.shape)
print(y_centers.shape)


#color = ['black', 'blue', 'purple', 'yellow', 'red', 'lime', 'cyan', 'white', 'orange', 'gray']
color = ['orange', 'blue', 'purple', 'yellow', 'red', 'lime']

fig = plt.figure()
ax = Axes3D(fig)
for i in range(len(x_centers)):
	ax.scatter(x_centers[i,0],x_centers[i,1],x_centers[i,2],s=50,c='black',alpha=1)
	cursor = Cursor(ax, useblit=True, color='black', linewidth=2)

for i in range(len(pca_points)):
	if(labels[i]<6):
		ax.scatter(pca_points[i,0],pca_points[i,1],pca_points[i,2],s=10,alpha=0.42,c=color[labels[i]])


plt.savefig('results/pca_cluster_t.png')
