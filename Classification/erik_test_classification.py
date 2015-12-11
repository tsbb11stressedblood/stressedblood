import cv2
import pylab
from whitecell import *
from classification import *
import math
from Segmentation import rbc_seg
from matplotlib.colors import ListedColormap
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics, preprocessing, cross_validation
from sklearn.utils import shuffle
import platform
import pickle # Needed for saving the trainer to file
from classer import training
from classer import feature_selection, classify_data

fff = open('../white_1.pik', 'r')
cell_list = pickle.load(fff)
testi = 0

#print len(cell_list)
#plt.figure()
#remove things we shouldn't train on:
del cell_list[43]
del cell_list[41]
del cell_list[20]
del cell_list[10]
#del cell_list[5]
del cell_list[3]
del cell_list[0]
#print len(cell_list)

del cell_list[36]

"""
for i,c in enumerate(cell_list[0:43]):
    cv2.drawContours(c.img, [c.contour], -1, [0,0,0], 2, offset=(-c.x,-c.y ))

    if i % 10 == 0:
        plt.figure()
        testi = 0
    plt.subplot(1,10,testi+1)
    plt.imshow(c.img)
    testi += 1
"""

"""
plt.figure()
plt.imshow(cell_list[0].img)
plt.figure()
plt.imshow(cell_list[1].img)
plt.figure()
plt.imshow(cell_list[2].img)
plt.show()
"""

#print cell_list[0].contour
def predict_cells(cell_list):
    WBC_data = []
    for cell in cell_list:
        wc = WhiteCell(cell, -1)
        wc_features = feature_selection(wc)
        WBC_data.append(wc_features)

    WBC_data= np.asarray(WBC_data)
    #prediction = classify_data(WBC_data, trainer)

    return WBC_data


#READ FROM IMAGE 2!
fff = open('../white_2.pik', 'r')
cell_list2 = pickle.load(fff)
testi = 0
"""
for i,c in enumerate(cell_list2):
    #cv2.drawContours(c.img, [c.contour], -1, [0,0,0], 2, offset=(-c.x,-c.y ))

    #if i % 10 == 0:
    #    plt.figure()
    #    testi = 0
    #plt.subplot(1,10,testi+1)
    #plt.imshow(c.img)
    #testi += 1
    window_width = int(c.img.shape[1] * 2)
    window_height = int(c.img.shape[0] * 2)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)
    cv2.imshow('dst_rt',cv2.cvtColor(c.img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
cv2.destroyAllWindows()
"""


#READ FROM ONLY_SMEARED!
fff = open('../only_smeared.pik', 'r')
cell_list_smeared = pickle.load(fff)


labels1 = [1,3,3,3,0,1,3,2,3,3,  1,3,1,3,1,0,3,0,3,3,  2,0,2,3,3,0,1,3,1,0,  0,1,1,0,1,0,1]
labels2 = [1,2,1,1,0,2,2,0,0,2,2,1,0,1,2,1,2,2,1,0,1,0,0,0,2,2,0,1,0,1]
labels_smeared = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]


feature_array = predict_cells(cell_list)
feature_array2 = predict_cells(cell_list2)
feature_array_smeared = predict_cells(cell_list_smeared)


#concatenate features
features_1_2 = np.append(feature_array,feature_array2, axis=0)
features = np.append(features_1_2, feature_array_smeared, axis=0)

#concatenate labels 1 and 2
labels = labels1+labels2+labels_smeared
trainer = training(features, labels)


fff = open('./trainer_easy.pik', 'w+')
pickle.dump(trainer, fff)