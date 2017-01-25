import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_images_from_folder(folder):
    """
    Loads an image from folder. Returns loaded images in an array.
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def filter_image(img, itr=10, temp=7, search=21):
    """
    Converts image to cielab. Filters the L-channel with means denoising. Returns the filtered image.
    """
    img_cielab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_l = img_cielab[:,:,0]
    img_filtered = cv2.fastNlMeansDenoising(img_l,None,itr,temp,search)
    return img_filtered

def cluster_image(img):
    """
    Uses k-means clustering to divide the image into K number of values.
    Returns the clustered image and its values.
    """
    img_reshape = img.reshape((img.shape[0] * img.shape[1], 1))
    img_reshape = np.float32(img_reshape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,values=cv2.kmeans(img_reshape,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    values = np.uint8(values)
    res = values[label.flatten()]
    res = res.reshape((img.shape))
    return res, values

def separate_image(img, values):
    """
    Separates the image into foreground and nuclei
    """
    values = sorted(values)
    mask = np.zeros((img.shape[0], img.shape[1]))
    foreground_value = values[1]
    nuclei_value = values[0]
    foreground = np.zeros((img.shape[0], img.shape[1]))
    nuclei = np.zeros((img.shape[0], img.shape[1]))
    foreground[img <= foreground_value] = 1
    nuclei[img <= nuclei_value] = 1
    return foreground, nuclei

def remove_edges_from_nuclei(img):
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    #img = cv2.erode(img,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    img = cv2.erode(img,kernel,iterations = 4)
    img = remove_objects(img,20)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img = cv2.dilate(img,kernel,iterations = 6)
    #img = cv2.dilate(img,kernel,iterations = 2)
    return img

def fill_foreground(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations = 3)
    return img

def perform_watershed(foreground, nuclei):
    ret, markers = cv2.connectedComponents(nuclei)
    foreground = 1 - foreground
    markers[markers > 0] +=1
    markers = foreground + markers
    foreground_reshape = np.zeros((foreground.shape[0], foreground.shape[1], 3), dtype=np.uint8)
    foreground_reshape[:,:,0] = 1-foreground
    foreground_reshape[:,:,1] = 1-foreground
    foreground_reshape[:,:,2] = 1-foreground
    markers_watershed = cv2.watershed(foreground_reshape,markers)

    return markers_watershed

def remove_objects(img,size):
    _,contours, hierarchy = cv2.findContours(img.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [con for con in contours if cv2.contourArea(con)>size]
    empty = np.zeros(img.shape)
    cv2.drawContours(empty, contours, -1, 255, -1)
    return empty