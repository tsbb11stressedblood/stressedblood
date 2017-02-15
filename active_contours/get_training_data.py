import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as pickle
from cell import Cell
from preprocessing import *
images,filenames = load_images_and_name_from_folder('../ground_truth')
#images = load_images_from_folder('../nice_areas/9W/')
split_cells = [11,43,71,72,88,107,128,139,151,152,157,180,208,215,231,239,254,311,329,367,372,399,437,444,470,499,504,
               519,525,530,551,570,575,590,611,622,646,659,666,694,695,704,763,785,795,798,805,818,827,836,858,867,
               907,929,955,971,975,1019,1024,1028,1046,1060,1120,1131,1143,1144,1157,1177,1189,1192,1216,1217,1234,1235,
               1287,1319,1336,1343,1348,1376,1397,1415,1456,1457,1469,1476,1507]
def cluster_image_two(img):
    """
    Uses k-means clustering to divide the image into K number of values.
    Returns the clustered image and its values.
    """
    img_reshape = img.reshape((img.shape[0] * img.shape[1], 1))
    img_reshape = np.float32(img_reshape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,values=cv2.kmeans(img_reshape,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    values = np.uint8(values)
    res = values[label.flatten()]
    res = res.reshape((img.shape))
    return res, values

def check_all_cells(images):
    for i,img in enumerate(images):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_filtered = filter_image(img)
        res, values = cluster_image_two(img_filtered)
        foreground, nuclei = separate_image(img_filtered, values)
        nuclei = remove_edges_from_nuclei(nuclei)
        foreground = fill_foreground(foreground)
        markers_watershed = perform_watershed(foreground.astype(np.uint8), nuclei.astype(np.uint8))
        img[markers_watershed == -1] = [255,0,0]
        print i
        plt.figure()
        plt.imshow(markers_watershed)
        plt.figure()
        plt.imshow(img)
        plt.show()

def check_split_cells(images, split_cells):
    for nr in split_cells:
        img = images[nr]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_filtered = filter_image(img)
        res, values = cluster_image_two(img_filtered)
        foreground, nuclei = separate_image(img_filtered, values)
        #plt.figure("foreground before")
        #plt.imshow(foreground)
        #plt.figure("nuclei before")
        #plt.imshow(nuclei)
        nuclei = remove_edges_from_nuclei(nuclei)
        foreground = fill_foreground(foreground)
        #plt.figure("foreground after")
        #plt.imshow(foreground)
        #plt.figure("nuclei after")
        #plt.imshow(nuclei)
        markers_watershed = perform_watershed(foreground.astype(np.uint8), nuclei.astype(np.uint8))
        img[markers_watershed == -1] = [255,0,0]
        print nr
        plt.figure()
        plt.imshow(img)
        plt.show()

def save_cells(images, filenames, list):
    training_cells = []
    for image_index,contour_index in enumerate(list):
        img = images[image_index]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_filtered = filter_image(img)
        res, values = cluster_image_two(img_filtered)
        foreground, nuclei = separate_image(img_filtered, values)
        nuclei = remove_edges_from_nuclei(nuclei)
        foreground = fill_foreground(foreground)
        markers_watershed = perform_watershed(foreground.astype(np.uint8), nuclei.astype(np.uint8))
        markers_watershed[markers_watershed != contour_index] = 0
        markers_watershed[markers_watershed == contour_index] = 1
        contour = cv2.findContours(markers_watershed.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cell = Cell(filenames[image_index],contour)
        training_cells.append(cell)
    return training_cells

check_all_cells(images)
