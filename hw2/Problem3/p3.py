import numpy as np
import cv2
import scipy.misc

image = cv2.imread('train-10/Mountain/image_0033.jpg')
surf = cv2.xfeatures2d.SURF_create()
kp, des = surf.detectAndCompute(image, None)
    
print(len(kp))
print(des.shape)


img2 = cv2.drawKeypoints(image, kp[:30], None, (255,0,0), 4)

scipy.misc.imsave('results/key_Mountain_0033.jpg', img2)
