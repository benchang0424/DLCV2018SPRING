# Problem2: Color and texture segmentation

## 2.(a) : Color segmentation  

Image | Original | RGB | LAB 
:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:
mountain|![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/mountain.jpg)  |  ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/results/p2_1/p2_m_rgb.jpg) | ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/results/p2_1/p2_m_lab.jpg)
zebra|![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/zebra.jpg)  |  ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/results/p2_1/p2_z_rgb.jpg) | ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/results/p2_1/p2_z_labttt.jpg)

## 2.(b) : Texture segmentation  

Image | Original | 38 dim. | 41 dim. (+LAB) 
:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:
mountain|![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/mountain.jpg)  |  ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/results/p2_2/m_38.jpg) | ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/results/p2_2/m_41.jpg)
zebra|![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/zebra.jpg)  |  ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/results/p2_2/z_38.jpg) | ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem2/results/p2_2/z_41.jpg)

&nbsp;

# Problem3: Recognition with bag of visual words
&nbsp;
## 3.(a) : Interest point detection (30 most dominant interest points detected)  
<div align=center><img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/key_Mountain_0033.jpg"><div align=left>

&nbsp;
## 3.(b) : Plot the visual words and the associated interest points in this PCA subspace.
<div align=center><img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/pca_cluster.png"><div align=left>

&nbsp;
## 3.(c) : Compute BoW of training images and plot their Hard-Sum, Soft-Sum, and Soft-Max, respectively.

Image | Hard-Sum | Soft-Sum | Soft-Max
:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:
Coast|![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Coast_image_0022.jpg_hardsum_hist.jpg)  |  ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Coast_image_0022.jpg_softsum_hist.jpg) | ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Coast_image_0022.jpg_softmax_hist_.jpg)
Forest|![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Forest_image_0020.jpg_hardsum_hist.jpg)  |  ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Forest_image_0020.jpg_softsum_hist.jpg) | ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Forest_image_0020.jpg_softmax_hist_.jpg)
Highway|![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Highway_image_0018.jpg_hardsum_hist.jpg)  |  ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Highway_image_0018.jpg_softsum_hist.jpg) | ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Highway_image_0018.jpg_softmax_hist_.jpg)
Mountain|![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Mountain_image_0044.jpg_hardsum_hist.jpg)  |  ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Mountain_image_0044.jpg_softsum_hist.jpg) | ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Mountain_image_0044.jpg_softmax_hist_.jpg)
Suburb|![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Suburb_image_0016.jpg_hardsum_hist.jpg)  |  ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Suburb_image_0016.jpg_softsum_hist.jpg) | ![](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw2/Problem3/results/p3_3_1000/Suburb_image_0016.jpg_softmax_hist_.jpg)