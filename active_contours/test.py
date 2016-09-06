import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import medpy.filter

def filter_image(img, itr=10, temp=7, search=21):
    #img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_cielab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_l = img_cielab[:,:,0]
    img_filtered = cv2.fastNlMeansDenoising(img_l,None,itr,temp,search)
    return img_filtered

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

def perform_watershed(foreground, nuclei, img):
    foreground = 1.0 - foreground
    ret, markers = cv2.connectedComponents(nuclei)
    markers = markers + foreground
    markers = cv2.watershed(img,markers)
    return markers

#img = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.06_1.png', 'r')
#img = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.06_2.png', 'r')
#img = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.17_1.png', 'r')
#img = mpimg.imread('../nice_areas/9W/512x512/2015-10-13 13.48_2.png', 'r')
img = mpimg.imread('../nice_areas/12W/2016-01-23 12.07_2.png', 'r')


img_filtered = filter_image(img)
img_clustered, values = cluster_image(img_filtered)
foreground, nuclei = separate_image(img_clustered, values)
nuclei = remove_edges_from_nuclei(nuclei)
foreground = fill_foreground(foreground)
#markers = perform_watershed(foreground, nuclei.astype(np.uint8), img)
#img[markers == -1] = [255,0,0]

#img_lap = cv2.Laplacian(img_1, cv2.CV_64F)

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow()
plt.figure()
#plt.imshow()
plt.figure()
#plt.imshow()
#print values
plt.show()
