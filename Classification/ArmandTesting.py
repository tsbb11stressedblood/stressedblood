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


WBCarray = np.load("WBC.npy")

LNE = LNE(WBCarray)

edges = cv2.Canny(WBCarray,100,200)

plt.imshow(edges)
plt.colorbar()
plt.show()

