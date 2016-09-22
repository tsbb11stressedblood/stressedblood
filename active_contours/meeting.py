import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import medpy.filter
def cluster_image(img):
    img_reshape = img.reshape((img.shape[0] * img.shape[1], 1))
    img_reshape = np.float32(img_reshape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(img_reshape,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img.shape))
    return res, center

def separate_image(img, values):
    mask = np.zeros((img.shape[0], img.shape[1]))
    foreground_value = values[1]
    nuclei_value = values[2]
    foreground = mask.copy()
    nuclei = mask.copy()
    foreground[img <= foreground_value] = 1
    nuclei[img <= nuclei_value] = 1
    return foreground, nuclei

def remove_edges_from_nuclei(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    img = cv2.erode(img,kernel,iterations = 1)
    return img

def fill_foreground(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations = 3)
    return img

img = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.06_1.png', 'r')
#img = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.06_2.png', 'r')
#img = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.17_1.png', 'r')
#img = mpimg.imread('../nice_areas/9W/512x512/2015-10-13 13.48_2.png', 'r')
#img = mpimg.imread('../nice_areas/12W/2016-01-23 12.07_2.png', 'r')

# Color transform
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_cielab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img_l = img_cielab[:,:,0]

# Show images
plt.figure("Original image")
plt.imshow(img)
plt.figure("Gray image")
plt.imshow(img_gray)
plt.figure("Cielab colorspace")
plt.imshow(img_l)
plt.show()

# Filter images
img_filtered = cv2.fastNlMeansDenoising(img_l,None,10,7,21)
img_gaussian = cv2.GaussianBlur(img_l, (7,7), 0)

# Show images
plt.figure("Original image")
plt.imshow(img)
plt.figure("Non-linear Means denoising")
plt.imshow(img_filtered)
plt.figure("Gaussian")
plt.imshow(img_gaussian)
plt.show()

# Cluster image
img_clustered, values = cluster_image(img_filtered)

# Show images
plt.figure("Non-linear Means denoising")
plt.imshow(img_filtered)
plt.figure("Clustered")
plt.imshow(img_clustered)
plt.show()

# Finding nuclei
foreground, nuclei = separate_image(img_clustered, values)
nuclei = remove_edges_from_nuclei(nuclei)
foreground = fill_foreground(foreground)
mask = np.zeros((img.shape[0], img.shape[1]))
mask[foreground == 1] = 1
mask[nuclei == 1] = 2


plt.figure("Nuclei")
plt.imshow(mask)
plt.figure("Original image")
plt.imshow(img)
plt.show()
