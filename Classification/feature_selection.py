
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


WBCarray = []
WBCarray_hsv = []
WBCarray_bw = []

for i in range(1, 21):
    WBCarray.append(np.load("white_" + str(i) + ".npy"));
    WBCarray_hsv.append( colorsys.convert_to_hsv( np.load("white_" + str(i) + ".npy") ) );
    WBCarray_bw.append(cv2.cvtColor(np.load("white_" + str(i) + ".npy"), cv2.COLOR_BGR2GRAY))


plt.figure();
for i in range(0, 20):
    tmp = WBCarray[i];
    #plt.figure();
    plt.subplot(5,4,i+1);
    plt.imshow(tmp[:,:,0])

plt.figure();
for i in range(0, 20):
    tmp = WBCarray[i];
    #plt.figure();
    plt.subplot(5,4,i+1);
    plt.imshow(tmp[:,:,1])

plt.figure();
for i in range(0, 20):
    tmp = WBCarray[i];
    #plt.figure();
    plt.subplot(5,4,i+1);
    plt.imshow(tmp[:,:,2])
    
#
# plt.figure();
# for i in range(0, 20):
#     tmp = WBCarray[i];
#     #plt.figure();
#     plt.subplot(5,4,i+1);
#     plt.imshow(tmp)
#
#
# plt.figure()
# for i in range(0, 20):
#     tmp = WBCarray[i]
#     #plt.figure();
#     plt.subplot(5,4,i+1)
#     plt.hist(tmp[:,:,0])
#     #plt.imshow(tmp[:,:,0])
# plt.figure()
#
# for i in range(0, 20):
#     tmp = WBCarray[i]
#     #plt.figure();
#     plt.subplot(5,4,i+1)
#     plt.hist(tmp[:,:,1])
#     #plt.imshow(tmp[:,:,1])
# plt.figure()
#
# for i in range(0, 20):
#     tmp = WBCarray[i]
#     #plt.figure();
#     plt.subplot(5,4,i+1)
#     plt.hist(tmp[:,:,2])
#     #plt.imshow(tmp[:,:,2])
#
# plt.figure()
# for i in range(0, 20):
#     tmp = WBCarray_hsv[i]
#     #plt.figure();
#     plt.subplot(5,4,i+1)
#     #plt.imshow(tmp[:,:,1])
#     plt.hist(tmp[:,:,0])
#
# plt.figure()
# for i in range(0, 20):
#     tmp = WBCarray_hsv[i]
#     #plt.figure();
#     plt.subplot(5,4,i+1)
#     #plt.imshow(tmp[:,:,1])
#     plt.hist(tmp[:,:,1])
#
#
# plt.figure()
# for i in range(0, 20):
#     tmp = WBCarray_hsv[i]
#     #plt.figure();
#     plt.subplot(5,4,i+1)
#     #plt.imshow(tmp[:,:,2])
#     plt.hist(tmp[:,:,2])

'''
plt.figure()
for i in range(0, 20):
    tmp = WBCarray_hsv[i]
    #plt.figure();
    plt.subplot(5,4,i+1)
    plt.imshow(tmp)

'''

plt.show()

