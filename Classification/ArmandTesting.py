"""
Armands leksak
"""

from Tkinter import *
import colorsys
import tkFileDialog
import tkMessageBox
from openslide import *
import os
from PIL import ImageTk as itk
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pylab
from Segmentation import rbc_seg



def convert_to_hsv(rgba_image):
    """
    Takes a RGBA image and return a HSVA image.
    The out put is of type float [0 1]
    :param rgba_image:
    :return: hsv_image
    """
    # Preprocessing
    shape_of_image = np.shape(rgba_image)
    rgba_image = rgba_image.astype(float)/255
    hsva = rgba_image

    for row in range(0,shape_of_image[0]):
        for column in range(0,shape_of_image[1]):
            hsva[row,column,0:3] = colorsys.rgb_to_hsv(rgba_image[row,column,0:3])


    hsva = hsva

    return hsva

def LNE(rgba_image):
    '''
    Enhances the nucleus through the green and saturation chanel
    :param rgba_image:
    :return: lne_image
    '''

    shape_of_image = np.shape(rgba_image)
    rgba_image = rgba_image.astype(float)
    hsva = convert_to_hsv(rgba_image)
    green = np.empty([shape_of_image[0], shape_of_image[1]])
    satur = np.empty([shape_of_image[0], shape_of_image[1]])
    lne_image = np.empty([shape_of_image[0], shape_of_image[1]])

    greenMax = np.amax(rgba_image[:,:,1])
    greenMin = np.amin(rgba_image[:,:,1])
    saturMax = np.amax(hsva[:,:,1])
    saturMin = np.amin(hsva[:,:,1])



    #Normelize the G and S chanel
    for row in range(0,shape_of_image[0]):
        for column in range(0,shape_of_image[1]):
            green[row,column] = (rgba_image[row,column,1])/(greenMax-greenMin)
            satur[row,column] = (hsva[row,column,1])/(saturMax-saturMin)
            lne_image[row,column] = satur[row,column]/green[row,column]


    return lne_image

def nuclues_fill(cellarray):  # Should take in only cell object
    '''
    Calculates the ratio of the nucleus. The input is the numpy array of one single cell
    :param cellArray:
    :return: fillratio
    '''
    hsv = convert_to_hsv(cellarray)
    lne = hsv[:,:,1]
    threshold = 0.75
    lne[lne>threshold*np.amax(lne)] = 1
    lne[lne<threshold*np.amax(lne)] = 0
    cellarea = 1000.0 # Remove when putting in a cell object
    nucleus_area = np.count_nonzero(lne)
    fillratio = nucleus_area/cellarea
    plt.figure(10)
    plt.imshow(lne)
    print type(cellarea)
    return fillratio

def erosion_test(cellarray):
    hsv = convert_to_hsv(cellarray)
    lne = hsv[:,:,1]
    threshold = 0.75
    lne[lne>threshold*np.amax(lne)] = 1
    lne[lne<threshold*np.amax(lne)] = 0

    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(lne, kernel, iterations=2)
    plt.figure(11)
    plt.subplot(221)
    plt.imshow(erosion)
    plt.subplot(222)
    plt.imshow(lne)



'''
WBCarray = np.load("WBC.npy")

LNE = LNE(WBCarray)

edges = cv2.Canny(WBCarray,100,200)
'''

WBCarray = np.load("lymphocyte.npy")
#erosion_test(WBCarray)
#list = rbc_seg.segmentation(WBCarray)
#Cell = 0
cellx = 66 #list[Cell].x
celly = 14 #list[Cell].y
cellw = 47 #list[Cell].w
cellh = 39 #list[Cell].h
cellarray = WBCarray[celly:(celly+cellh),cellx:(cellx+cellw),:]
#cellarray = WBCarray;

hsv = convert_to_hsv(cellarray)
lne = hsv[:,:,1]
lne_th = hsv[:,:,1].copy()      # Fucking deep copy


#print nuclues_fill(cellarray)

#imgray = cv2.cvtColor(cellarray, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#mask = np.zeros_like(thresh)
#cv2.drawContours(mask, contours, 0, 255, -1)
#out = np.zeros_like(thresh)


plt.figure(2)
plt.subplot(221)
plt.imshow(lne)
plt.colorbar()
plt.subplot(222)
plt.imshow(lne_th)
plt.colorbar()
plt.subplot(223)
plt.imshow(cellarray)
plt.colorbar()
plt.show()

