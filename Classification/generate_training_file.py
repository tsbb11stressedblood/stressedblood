import cv2
import pylab
from whitecell import *
from Segmentation import rbc_seg
import numpy as np
import matplotlib.pyplot as plt
import platform
import pickle
import classer
from Segmentation import Klara_test

def get_feature_array(cell_list):
    WBC_data = []
    for cell in cell_list:
        wc = WhiteCell(cell, -1)
        wc_features = classer.feature_selection(wc)
        WBC_data.append(wc_features)

    WBC_data= np.asarray(WBC_data)
    #prediction = classify_data(WBC_data, trainer)

    return WBC_data

training_files = ["only_whites_clean.png", "only_whites_2.png", "only_whites_1_hard.png", "only_smeared.png"]

labels2 = [1,2,1,1,0,2,2,0,0,2,2,1,0,1,2,1,2,2,1,0,1,0,0,0,2,2,0,1,0,1]
labels_smeared = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
labels1 = [2,2,3,0,1,2,2,2,1,1,1,3,0,2,0,2,3,0,1,1, 0,0,1,1,0,1,0,1,3,3, 0,1]
labels_1_hard = [2,1,0,0,1,0,2,2,1,2,1,3,2,2,1,1,1,2,2,0,0,2,0,2,0,2,2,2,0,0]
#labels_others_1 = [3 for i in range(len(cell_list_others_1))]

all_labels = [labels1, labels2, labels_1_hard, labels_smeared]
labels = []
for i,filename in enumerate(training_files):
        testimg = cv2.imread("../npyimages/" + filename)
        testimg = cv2.cvtColor(testimg, cv2.COLOR_BGR2RGB)
        #cell_list = rbc_seg.segmentation(testimg)
        #testimg = np.load("../Classification/others_2.npy")
        #plt.imshow(testimg)
        #plt.show()
        if filename == "only_whites_clean.png":
            cell_list = rbc_seg.segmentation(testimg)
            del cell_list[11]
            del cell_list[7]
            print "len clean:", len(cell_list)

        if filename == "only_whites_2.png":
            cell_list = rbc_seg.segmentation(testimg)
        if filename == "only_whites_1_hard.png":
            cytoplasm_cont, nuclei_mask, removed_cells, exchanged_cells = Klara_test.cell_watershed(testimg)
            cell_list_1_hard_ = Klara_test.modify_cell_list(testimg, cytoplasm_cont, nuclei_mask)
            cell_list = []
            cell_list.append(cell_list_1_hard_[8])
            cell_list.append(cell_list_1_hard_[10])
            cell_list.append(cell_list_1_hard_[16])
            cell_list.append(cell_list_1_hard_[23])
            cell_list.append(cell_list_1_hard_[32])
            cell_list.append(cell_list_1_hard_[47])
            cell_list.append(cell_list_1_hard_[59])
            cell_list.append(cell_list_1_hard_[62])
            cell_list.append(cell_list_1_hard_[69])
            cell_list.append(cell_list_1_hard_[87])
            cell_list.append(cell_list_1_hard_[89])
            cell_list.append(cell_list_1_hard_[93])
            cell_list.append(cell_list_1_hard_[97])
            cell_list.append(cell_list_1_hard_[98])
            cell_list.append(cell_list_1_hard_[99])
            cell_list.append(cell_list_1_hard_[101])
            cell_list.append(cell_list_1_hard_[102])
            cell_list.append(cell_list_1_hard_[104])
            cell_list.append(cell_list_1_hard_[105])
            cell_list.append(cell_list_1_hard_[106])
            cell_list.append(cell_list_1_hard_[108])
            cell_list.append(cell_list_1_hard_[109])
            cell_list.append(cell_list_1_hard_[110])
            cell_list.append(cell_list_1_hard_[112])
            cell_list.append(cell_list_1_hard_[113])
            cell_list.append(cell_list_1_hard_[114])
            cell_list.append(cell_list_1_hard_[115])
            cell_list.append(cell_list_1_hard_[118])
            cell_list.append(cell_list_1_hard_[119])
            cell_list.append(cell_list_1_hard_[120])
            print "LEEN:", len(cell_list)
        else:
            cell_list = rbc_seg.segmentation(testimg)
            print "other:", len(cell_list)

        feature_array = get_feature_array(cell_list)
        if i==0:
            features = feature_array
        print feature_array.shape
        print features.shape
        features = np.append(features, feature_array, axis=0)
        print features.shape
        labels = labels + all_labels[i]

labels = labels + [3 for i in range(32)]

#labels = labels1+labels2+labels_smeared + labels_others_1 + labels_1_hard
print features.shape, len(labels)
trainer = classer.training(features, labels)

fff = open('./trainer_easy.pik', 'w+')
pickle.dump(trainer, fff)