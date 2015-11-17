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

    return fillratio

def erosion_test(cellarray):
    hsv = convert_to_hsv(cellarray)
    lne = hsv[:,:,1]
    threshold = 0.7
    lne[lne>threshold*np.amax(lne)] = 1
    lne[lne<threshold*np.amax(lne)] = 0

    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(lne, kernel, iterations=3)

    ret, markers = cv2.connectedComponents(np.uint8(erosion))
    ret = ret-1
    print markers, ret

    plt.figure(11)
    plt.subplot(221)
    plt.imshow(markers)
    plt.subplot(222)
    plt.imshow(lne)


'''
WBCarray = np.load("WBC.npy")

LNE = LNE(WBCarray)

edges = cv2.Canny(WBCarray,100,200)
'''

WBCarray = np.load("white_1.npy")
#erosion_test(WBCarray)

#list = rbc_seg.segmentation(WBCarray)
#Cell = 0
#cellx = 66 #list[Cell].x
#celly = 14 #list[Cell].y
#cellw = 47 #list[Cell].w
#cellh = 39 #list[Cell].h
#cellarray = WBCarray[celly:(celly+cellh),cellx:(cellx+cellw),:]
cellarray = WBCarray;

def get_mean_var(cellarray):
    mean_array = []
    var_array = []
    hsv = convert_to_hsv(cellarray)
    for i in range(0,3):
        lne = hsv[:,:,i]
        mean_array.append(np.mean(lne))
        var_array.append(np.var(lne))

    for i in range(0,3):
        lne = cellarray[:,:,i]/255.0
        mean_array.append(np.mean(lne))
        var_array.append(np.var(lne))

    return mean_array, var_array

print get_mean_var(WBCarray)

def get_energy(cellarray):
    energy_array = []
    hsv = convert_to_hsv(cellarray)
    for i in range(0,3):
        lne = hsv[:,:,i]
        energy_array.append(np.sum(np.power(lne, 2))/np.size(cellarray))

    for i in range(0,3):
        lne = cellarray[:,:,i]/255.0
        energy_array.append(np.sum(np.power(lne, 2))/np.size(cellarray))

    return energy_array

print get_energy(WBCarray)

'''
#plt.bar(bin_edges[:-1], hist, width = 0.05)
plt.figure()
for i in range(0, 20):
    WBCarray = np.load("white_" + str(i + 1) + ".npy" )
    WBC_hsv = convert_to_hsv(WBCarray)
    color = ('b', 'g', 'r')
    plt.subplot(5,4,i+1)
    for j,col in enumerate(color):
        WBC_h = WBC_hsv[:,:,j]
        hist,bin_edges = np.histogram(WBC_h, 50)
        plt.plot(hist, color= col)
        #plt.xlim([0,256])

plt.show()
'''
'''
WBCarray = np.load("white_8.npy")
plt.figure()
for i in range(0, 20):
    WBCarray = np.load("white_" + str(i + 1) + ".npy" )
    WBC_bw = cv2.cvtColor(WBCarray, cv2.COLOR_BGR2GRAY)
    plt.subplot(5,4,i+1)
    hist,bin_edges = np.histogram(WBC_bw, 5)
    print hist
    plt.plot(hist)
    plt.xlim([0,5])
    plt.subplot(5,4,i+1)
plt.show()
'''

WBCarray = np.load("white_8.npy")

def bw_histogram(cellarray, bin):
    cellarray_bw = cv2.cvtColor(cellarray, cv2.COLOR_BGR2GRAY)
    hist,bin_edges = np.histogram(cellarray_bw, bin)
    bin_index = np.argmax(hist)
    bin_value = hist[bin_index]
    im_size = float(np.size(cellarray_bw))
    bin_perc = bin_value/im_size
    return bin_perc, bin_index/float(bin)


print bw_histogram(WBCarray, 30)

def hsv_histogram(cellarray, bin):
    cellarray_hsv = convert_to_hsv(cellarray)
    im_size = float(np.size(cellarray_hsv))
    perc_array = []
    index_array = []
    for i in range(0,3):
        cellarray_h = cellarray_hsv[:,:,i]
        hist,bin_edges = np.histogram(cellarray_h, bin)
        bin_index = np.argmax(hist)
        bin_value = hist[bin_index]
        bin_perc = bin_value/im_size
        perc_array.append(bin_perc)
        index_array.append(bin_index/float(bin))
    return index_array, perc_array

def rgb_histogram(cellarray, bin):
    im_size = float(np.size(cellarray))
    perc_array = []
    index_array = []
    for i in range(0,3):
        cellarray_h = cellarray[:,:,i]
        hist,bin_edges = np.histogram(cellarray_h, bin)
        bin_index = np.argmax(hist)
        bin_value = hist[bin_index]
        bin_perc = bin_value/im_size
        perc_array.append(bin_perc)
        index_array.append(bin_index/float(bin))
    return index_array, perc_array

print rgb_histogram(WBCarray, 30)

print hsv_histogram(WBCarray, 30)



'''
WBCarray = np.load("white_8.npy")
plt.figure()
for i in range(0, 20):
    WBCarray = np.load("white_" + str(i + 1) + ".npy" )
    color = ('b', 'g', 'r')
    plt.subplot(5,4,i+1)

    for i,col in enumerate(color):
        histr = cv2.calcHist([WBCarray], [i], None, [10], [0,256])
        plt.plot(histr, color= col)
        plt.xlim([0,10])
    plt.subplot(5,4,i+1)
plt.show()

'''


#print nuclues_fill(cellarray)

#imgray = cv2.cvtColor(cellarray, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#mask = np.zeros_like(thresh)
#cv2.drawContours(mask, contours, 0, 255, -1)
#out = np.zeros_like(thresh)

'''
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


WBC_array = np.load('simple_test.npy')
list = rbc_seg.segmentation(WBC_array)
WBC_img = list[0].img
plt.figure()
plt.imshow(WBC_img)
plt.show()
'''
