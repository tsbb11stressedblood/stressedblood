from skimage.feature import local_binary_pattern
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
import copy
from ColourNM import spacetransformer


WBCarray = []
WBCarray_hsv = []
WBCarray_bw = []

for i in range(1, 21):
    #WBCarray.append(np.load("white_" + str(i) + ".npy"));
    WBCarray_hsv.append( cv2.cvtColor(np.load("white_" + str(i) + ".npy"), cv2.COLOR_BGR2HSV) );
    #WBCarray_bw.append(cv2.cvtColor(np.load("white_" + str(i) + ".npy"), cv2.COLOR_BGR2GRAY))
    WBCarray_bw.append(spacetransformer.im2c(np.load("white_" + str(i) + ".npy"), 8))

WBC_masked = []
for i in range(0,20):
    #plt.figure(1)
    test_img = copy.copy(WBCarray_bw[i])
    nrows, ncols = test_img.shape
    row, col = np.ogrid[:nrows, :ncols]
    cnt_row, cnt_col = nrows / 2, ncols / 2
    outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 < (nrows / 2)**2)
    #test_img[outer_disk_mask==0] = -1
    test_img[test_img<0.3] = -1
    test_img[outer_disk_mask==0] = -1
    WBC_masked.append(test_img)
    #plt.subplot(5,4,i+1)
    #plt.imshow(test_img)
    #plt.figure(2)
    #plt.subplot(5,4,i+1)
    #plt.imshow(WBCarray_bw[i])

# 8, 2, 7
# 8: 15 features
# 2: 10 features
# 7: 10 features between 0-0.2
WBC_color = []
WBC_color = []
for i in range(0, 20):
    WBC_color.append(spacetransformer.im2c(np.load("white_" + str(i + 1) + ".npy"), 8))
    #WBC_color.append((cv2.cvtColor(np.load("white_" + str(i+1) + ".npy"), cv2.COLOR_BGR2GRAY))/255.0)
    plt.figure(1)
    #lbp = local_binary_pattern(WBC_color[i], 24, 3)
    lbp = WBC_color[i]
    mask = WBC_masked[i]
    lbp[mask == -1] = -1
    hist,_ = np.histogram(lbp,15, [0,1])
    hist = hist/float(np.size(lbp[lbp>0]))
    plt.subplot(5,4,i+1);
    plt.imshow(lbp)
    plt.figure(2)
    plt.subplot(5,4,i+1);
    #plt.ylim([0,1])
    plt.plot(hist)

plt.show()

# def erosion_test():
#     i = 4
#     hsv = WBCarray_hsv[i]
#     mask = WBC_masked[i]
#     lne = hsv[:,:,1]
#     lne[mask == -1] = -1
#     threshold = 0.7
#     lne[lne>threshold*np.amax(lne)] = 1
#     lne[lne<threshold*np.amax(lne)] = 0
#
#     kernel = np.ones((2,2), np.uint8)
#     erosion = cv2.erode(lne, kernel, iterations=3)
#
#     ret, markers = cv2.connectedComponents(np.uint8(erosion))
#     ret = ret-1
#     print markers, ret
#
#     plt.figure(11)
#     plt.subplot(221)
#     plt.imshow(markers)
#     plt.subplot(222)
#     plt.imshow(lne)
#
# erosion_test()
# plt.show()