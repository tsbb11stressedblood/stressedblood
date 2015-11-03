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
<<<<<<< Updated upstream
=======
from Segmentation import rbc_seg
>>>>>>> Stashed changes


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

<<<<<<< Updated upstream

WBCarray = np.load("WBC.npy")

LNE = LNE(WBCarray)

edges = cv2.Canny(WBCarray,100,200)

plt.imshow(edges)
plt.colorbar()
=======
WBCarray = np.load("lymphocyte.npy")

LNE = LNE(WBCarray)

LNE_max = np.amax(LNE)*0.5
print LNE_max
LNE[LNE <= LNE_max] = -1;

LNE[LNE > LNE_max] = 1;

hsv = convert_to_hsv(WBCarray)
edges = cv2.Canny(WBCarray,100,200)

#plt.imshow(edges)
plt.figure(1)


plt.subplot(231)
plt.imshow(WBCarray[:,:,0])
plt.subplot(232)
plt.imshow(WBCarray[:,:,1])
plt.subplot(233)
plt.imshow(WBCarray[:,:,2])

plt.subplot(234)
plt.imshow(hsv[:,:,0])
plt.subplot(235)
plt.imshow(hsv[:,:,1])
plt.subplot(236)
plt.imshow(hsv[:,:,2])


>>>>>>> Stashed changes
plt.show()

