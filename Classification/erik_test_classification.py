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
"""
print len(cell_list)
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

del cell_list[11]
del cell_list[7]
"""
for i,c in enumerate(cell_list[0:43]):
#cv2.drawContours(c.img, [c.contour], -1, [0,0,0], 2, offset=(-c.x,-c.y ))

    if i % 10 == 0:
        plt.figure()
        testi = 0
    plt.subplot(1,10,testi+1)
    plt.imshow(c.img)
    testi += 1

plt.show()
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
"""
for i,c in enumerate(cell_list):
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


#READ FROM HARD IMAGE 1!
fff = open('../only_whites_1_hard.pik', 'r')
cell_list_1_hard_ = pickle.load(fff)
testi = 0

cell_list_1_hard = []
cell_list_1_hard.append(cell_list_1_hard_[8])
cell_list_1_hard.append(cell_list_1_hard_[10])
cell_list_1_hard.append(cell_list_1_hard_[16])
cell_list_1_hard.append(cell_list_1_hard_[23])
cell_list_1_hard.append(cell_list_1_hard_[32])
cell_list_1_hard.append(cell_list_1_hard_[47])
cell_list_1_hard.append(cell_list_1_hard_[59])
cell_list_1_hard.append(cell_list_1_hard_[62])
cell_list_1_hard.append(cell_list_1_hard_[69])
cell_list_1_hard.append(cell_list_1_hard_[87])
cell_list_1_hard.append(cell_list_1_hard_[89])
cell_list_1_hard.append(cell_list_1_hard_[93])
cell_list_1_hard.append(cell_list_1_hard_[97])
cell_list_1_hard.append(cell_list_1_hard_[98])
cell_list_1_hard.append(cell_list_1_hard_[99])
cell_list_1_hard.append(cell_list_1_hard_[101])
cell_list_1_hard.append(cell_list_1_hard_[102])
cell_list_1_hard.append(cell_list_1_hard_[104])
cell_list_1_hard.append(cell_list_1_hard_[105])
cell_list_1_hard.append(cell_list_1_hard_[106])
cell_list_1_hard.append(cell_list_1_hard_[108])
cell_list_1_hard.append(cell_list_1_hard_[109])
cell_list_1_hard.append(cell_list_1_hard_[110])
cell_list_1_hard.append(cell_list_1_hard_[112])
cell_list_1_hard.append(cell_list_1_hard_[113])
cell_list_1_hard.append(cell_list_1_hard_[114])
cell_list_1_hard.append(cell_list_1_hard_[115])
cell_list_1_hard.append(cell_list_1_hard_[118])
cell_list_1_hard.append(cell_list_1_hard_[119])
cell_list_1_hard.append(cell_list_1_hard_[120])


for i,c in enumerate(cell_list_1_hard):
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




#Read from others_1!!
fff = open('../others_1.pik', 'r')
cell_list_others_1 = pickle.load(fff)
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

#labels1 = [1,3,3,3,0,1,3,2,3,3,  1,3,1,3,1,0,3,0,3,3,  2,0,2,3,3,0,1,3,1,0,  0,1,1,0,1,0,1]
labels2 = [1,2,1,1,0,2,2,0,0,2,2,1,0,1,2,1,2,2,1,0,1,0,0,0,2,2,0,1,0,1]
labels_smeared = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

labels1 = [2,2,3,0,1,2,2,2,1,1,1,3,0,2,0,2,3,0,1,1, 0,0,1,1,0,1,0,1,3,3, 0,1]

labels_1_hard = [2,1,0,0,1,0,2,2,1,2,1,3,2,2,1,1,1,2,2,0,0,2,0,2,0,2,2,2,0,0]


#66 3's
labels_others_1 = [3 for i in range(len(cell_list_others_1))]

print len(labels1)
print len(labels1) + len(labels2) + len(labels_smeared)

feature_array = predict_cells(cell_list)
feature_array2 = predict_cells(cell_list2)
feature_array_smeared = predict_cells(cell_list_smeared)
feature_array_others_1 = predict_cells(cell_list_others_1)
feature_array_1_hard = predict_cells(cell_list_1_hard)

print feature_array.shape

#concatenate features
features_1_2 = np.append(feature_array,feature_array2, axis=0)
features = np.append(features_1_2, feature_array_smeared, axis=0)
features = np.append(features, feature_array_others_1, axis=0)
features = np.append(features, feature_array_1_hard, axis=0)


#concatenate labels 1 and 2
labels = labels1+labels2+labels_smeared + labels_others_1 + labels_1_hard
trainer = training(features, labels)


fff = open('./trainer_easy.pik', 'w+')
pickle.dump(trainer, fff)