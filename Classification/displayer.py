import classer as cl
import cv2
import pylab
from whitecell import *
from classification import *
from Segmentation import rbc_seg
import itertools
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics

for i in range(1, 21):
    WBC_array = np.load("white_" + str(i) + ".npy")
    plt.figure(1)
    plt.imshow(WBC_array)
    plt.title("white_" + str(i) + ".npy")
    plt.show(1)



