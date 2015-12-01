#import classer as cl
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
import platform
import pickle # Needed for saving the trainer to file
#import classer


class WhiteCell:
    def __init__(self, cell, _label):
        self.img = cell    # Remove img to use one white_
        self.size = float(np.size(self.img))
        self.make_mask()       # Comment out to use one white_
        #self.mask = cell.mask

    def make_mask(self):
        mask = copy.copy(self.img)
        mask = spacetransformer.im2c(mask, 8)
        nrows, ncols = mask.shape
        row, col = np.ogrid[:nrows, :ncols]
        cnt_row, cnt_col = nrows / 2, ncols / 2
        outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 < (nrows / 2)**2)
        mask[mask<0.3] = -1
        mask[outer_disk_mask==0] = -1
        self.mask = mask

    def get_mean_var_energy(self, color):
        col = spacetransformer.im2c(self.img, color)
        energy = np.sum(np.power(col, 2))/self.size
        return [np.mean(col), np.var(col), energy]

    def hist_features(self, color, bins, interval = [0,1]):
        col = spacetransformer.im2c(self.img, color)
        col[self.mask == -1] = -1
        hist,_ = np.histogram(col,bins,interval)
        hist = hist/float(np.size(col[col>=0]))
        return hist

def feature_selection(whitecell):
    feature_array = []
    feature_array.append(whitecell.get_mean_var_energy(2))
    feature_array.append(whitecell.get_mean_var_energy(7))
    feature_array.append(whitecell.get_mean_var_energy(8))
    feature_array.append(whitecell.hist_features(2,10))
    feature_array.append(whitecell.hist_features(7,10,[0,0.2]))
    feature_array.append(whitecell.hist_features(8,15))
    return [item for sublist in feature_array for item in sublist]

def reduce_dimension(features, keep):
    '''
    Row = sample
    Column = features

    :param features:
    :param keep:
    :return: reduced_features
    '''

    features_shape = np.shape(features)

    if keep > features_shape[1]:
        print 'Error, not enough features'
    else:
        pca = PCA(n_components = keep)
        reduced_features = pca.fit_transform(features)

        return reduced_features


def training(training_features, training_labels):
    '''
    Train
    row = sample
    column = feature
    Test
    Vector with lables
    :param training_features:
    :param training_labels:
    :return: trained_classifier
    '''
    C = 1.0 # SVM regularization parameter
    #svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    trained_classifier = svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(training_features, training_labels)
    #poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    #lin_svc = svm.LinearSVC(C=C).fit(X, y)

    #trained_classifier = svm.SVC(kernel='poly', degree=3, C=C).fit(training_features, training_labels)
    #trained_classifier = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(training_features, training_labels)
    trained_classifier = svm.LinearSVC(C=C).fit(training_features, training_labels)
    #trained_classifier = svm.SVC(kernel='poly', degree=3, C=C).fit(training_features, training_labels)
    return trained_classifier


def classify_data(features,trained_classifier):
    '''
    :param features:
    :param trained_classifier:
    :return: predicted_classes
    '''
    predicted_classes  = trained_classifier.predict(features)

    return predicted_classes

## FROM white_
WBC_data = []
for i in range(1, 21):
 WBC_array = np.load("white_" + str(i) + ".npy")
 wc = WhiteCell(WBC_array, -1)
 wc_features = feature_selection(wc)
 WBC_data.append(wc_features)

WBC_data= np.asarray(WBC_data)
np.save("white_feature_array.npy",WBC_data )

# FROM CELL-LIST
#RBC_array = np.load("../gui/red_shit.npy")
#REB_celllist=rbc_seg.segmentation(RBC_array)


#RBC_data = []
#for cell in REB_celllist:
# wc = WhiteCell(cell, -1)
# wc_features = feature_selection(wc)
# RBC_data.append(wc_features)


#RBC_data= np.asarray(RBC_data)
#np.save("red_feature_array.npy",RBC_data )

WBC_data = np.load("white_feature_array.npy")
#RBC_data = np.load("red_feature_array.npy")

#feature_array = np.append(WBC_data,RBC_data[0:19],axis=0)
feature_array = WBC_data

WBC_labels = np.array([0,1,1,0,1,2,1,0,1,0,0,0,2,2,0,1,1,2,1,1])
#RBC_labels=3*np.ones([np.shape(RBC_data[0:19])[0]])

#labels_array = np.append(WBC_labels,RBC_labels,axis=0)
labels_array = WBC_labels

print np.shape(labels_array)
print labels_array

X = feature_array

y = labels_array

trainer = training(X, y)

if platform.system() == "Windows":
    filename = "trainer_win.pik"
else:
    filename = "trainer.pik"

with open(filename, "wb") as f:
    pickle.dump(trainer, f)

with open(filename, 'rb') as f:
    trainerTest = pickle.load(f)

prediction = classify_data(X,trainerTest)
confusion = metrics.confusion_matrix(y, prediction)

print confusion
