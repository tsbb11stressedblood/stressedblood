#import classer as cl
import cv2
import pylab
from whitecell import *
from classification import *
import math
from Segmentation import rbc_seg
import itertools
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics, preprocessing, cross_validation
from sklearn.utils import shuffle
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
    #trained_classifier = svm.SVC(kernel='rbf', gamma=0.2, C=C).fit(training_features, training_labels)
    trained_classifier = svm.LinearSVC(C=C).fit(training_features, training_labels)

    return trained_classifier


def classify_data(features,trained_classifier):
    '''
    :param features:
    :param trained_classifier:
    :return: predicted_classes
    '''
    predicted_classes  = trained_classifier.predict(features)

    return predicted_classes

def cross_validation(features, labels):
    '''
    Calculate the cross accuracy for 4 sets from the data
    :param features:
    :param lables:
    :param sets:
    :return: accuracy
    '''

    features, labels = shuffle(features, labels, random_state=1) #consistently shuffle data
    sets = 4.0
    num_samples = np.shape(features)[0]
    set_samples = math.floor(float(num_samples)/sets)

    # Get each set of training and test data
    training_data_1 = features[0:3*set_samples,:]
    training_lables_1 = labels[0:3*set_samples]
    test_data_1 = features[3*set_samples:num_samples,:]
    test_lables_1 = labels[3*set_samples:num_samples]
    training_data_2 = np.append(features[0:2*set_samples,:], features[3*set_samples:num_samples,:], axis=0)
    training_lables_2 = np.append(labels[0:2*set_samples], labels[3*set_samples:num_samples], axis=0)
    test_data_2 = features[2*set_samples:3*set_samples,:]
    test_lables_2 = labels[2*set_samples:3*set_samples]
    training_data_3 = np.append(features[0:1*set_samples,:], features[2*set_samples:num_samples,:], axis=0)
    training_lables_3 = np.append(labels[0:1*set_samples], labels[2*set_samples:num_samples], axis=0)
    test_data_3 = features[1*set_samples:2*set_samples,:]
    test_lables_3 = labels[1*set_samples:2*set_samples]
    training_data_4 = features[1:num_samples,:]
    training_lables_4 = labels[1:num_samples]
    test_data_4 = features[0*set_samples:1*set_samples,:]
    test_lables_4 = labels[0*set_samples:1*set_samples]

    # Calculate accuracy for each set
    total_accuracy =[]

    trainer_1 = training(training_data_1, training_lables_1)
    prediction = classify_data(test_data_1,trainer_1)
    confusion = metrics.confusion_matrix(test_lables_1, prediction)
    print"Confusion matrix 1:", "\n" ,confusion
    accuracy = float(np.trace(confusion))/float(np.sum(confusion))
    print "Accuracy:", accuracy, "\n"
    total_accuracy.append(accuracy)

    trainer_2 = training(training_data_2, training_lables_2)
    prediction = classify_data(test_data_2,trainer_2)
    confusion = metrics.confusion_matrix(test_lables_2, prediction)
    print"Confusion matrix 2:", "\n" ,confusion
    accuracy = float(np.trace(confusion))/float(np.sum(confusion))
    print "Accuracy:", accuracy, "\n"
    total_accuracy.append(accuracy)

    trainer_3 = training(training_data_3, training_lables_3)
    prediction = classify_data(test_data_3,trainer_3)
    confusion = metrics.confusion_matrix(test_lables_3, prediction)
    print"Confusion matrix 3:", "\n" ,confusion
    accuracy = float(np.trace(confusion))/float(np.sum(confusion))
    print "Accuracy:", accuracy, "\n"
    total_accuracy.append(accuracy)

    trainer_4 = training(training_data_4, training_lables_4)
    prediction = classify_data(test_data_4,trainer_4)
    confusion = metrics.confusion_matrix(test_lables_4, prediction)
    print"Confusion matrix 4:", "\n" ,confusion
    accuracy = float(np.trace(confusion))/float(np.sum(confusion))
    print "Accuracy:", accuracy, "\n"
    total_accuracy.append(accuracy)

    print "All accuracys",total_accuracy
    total_accuracy =np.asarray(total_accuracy)

    print("Accuracy: %0.3f (+/- %0.3f)" % (total_accuracy.mean(), total_accuracy.std() * 2))

    return total_accuracy


## FROM white_
# WBC_data = []
# for i in range(1, 101):
#  WBC_array = np.load("white_" + str(i) + ".npy")
#  wc = WhiteCell(WBC_array, -1)
#  wc_features = feature_selection(wc)
#  WBC_data.append(wc_features)
#
# WBC_data= np.asarray(WBC_data)
# np.save("white_feature_array.npy",WBC_data )

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


#WBC_labels = np.array([0,1,1,0,1,2,1,0,1,0,0,0,2,2,0,1,1,2,1,1,1,2,2,0,2,1,2,2,1,2,1,2,2,2,2,1,2,2,1,1,2,1,1,1,1,0,2,1,1,1,2,1,1,1,2,2,2,2,0,2,1,1,1,0,2,2,2,2,1,2,2,1,0,1,1,2,2,2,2,2,1,2,1,2,0,1,2,2,1,1,2,0,2,2,0,1,1,1,2,0])
#np.save("WBC_labels.npy",WBC_labels)
WBC_labels = np.load("WBC_labels.npy")

#RBC_labels=3*np.ones([np.shape(RBC_data[0:19])[0]])
#labels_array = np.append(WBC_labels,RBC_labels,axis=0)


X = feature_array
y = WBC_labels
trainer = training(X, y)

if platform.system() == "Windows":
    filename = "trainer_win.pik"
else:
    filename = "trainer.pik"

with open(filename, "wb") as f:
    pickle.dump(trainer, f)

with open(filename, 'rb') as f:
    trainer = pickle.load(f)

# prediction = classify_data(test_data,trainer)
# confusion = metrics.confusion_matrix(test_labels, prediction)
#
# print"Confusion matrix:", "\n" ,confusion, "\n"
# accuracy = float(np.trace(confusion))/float(np.sum(confusion))
# print "Accuracy:", accuracy

total_accuracy = cross_validation(feature_array,WBC_labels)
