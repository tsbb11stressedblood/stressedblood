import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import medpy.filter

def filter_image(img, itr=10, temp=7, search=21):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_cielab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_l = img_cielab[:,:,0]
    img_filtered = cv2.fastNlMeansDenoising(img_l,None,itr,temp,search)
    return img_filtered

def cluster_image(img):
    # K-means cluster
    img_reshape = img.reshape((img.shape[0] * img.shape[1], 1))
    img_reshape = np.float32(img_reshape)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(img_reshape,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img.shape))
    return res

#img = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.06_1.png', 'r')
img = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.06_2.png', 'r')
#img = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.17_1.png', 'r')

img_1 = filter_image(img)
img_2 = cluster_image(img_1)

img_nuc = np.zeros((img.shape[0], img.shape[1]))
img_nuc[img_2<100] = 1
kernel = np.ones((5,5),np.uint8)
img_erode = cv2.erode(img_nuc,kernel,iterations = 2)
#img_dilate = cv2.dilate(img_erode,kernel,iterations = 2)
img_dilate = img_erode
img_2[img_2<100] = 149
img_2 = img_dilate*50 + img_2

img_lap = cv2.Laplacian(img_1, cv2.CV_64F)
plt.figure()
plt.imshow(img_2)
plt.figure()
plt.imshow(img_1)
plt.figure()
plt.imshow(cluster_image(img_1))
plt.figure()
plt.imshow(img_lap)

plt.show()
