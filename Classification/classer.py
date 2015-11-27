import cv2
import pylab
from whitecell import *
from Segmentation import rbc_seg
import itertools
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
import platform
import pickle

iris = datasets.load_iris()

# Trainer is saved to disk, import it to use it in classification.
if platform.system() == "Windows":
    filename = "trainer_win.pik"
else:
    filename = "trainer.pik"

with open("../Classification/" + filename, 'rb') as f:
    trainer = pickle.load(f)


def predict_cells(cell_list):
    WBC_data = []
    print len(cell_list)
    for cell in cell_list:
        wc = WhiteCell(cell, -1)
        wc_features = feature_selection(wc)
        WBC_data.append(wc_features)

    WBC_data= np.asarray(WBC_data)
    prediction = classify_data(WBC_data, trainer)

    return prediction


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













