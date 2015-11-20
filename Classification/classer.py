import cv2
import pylab
from whitecell import *
from Segmentation import rbc_seg
import itertools
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics

iris = datasets.load_iris()
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

    #svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    #rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    #poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    #lin_svc = svm.LinearSVC(C=C).fit(X, y)

    C = 1 # SVM regularization parameter
    #trained_classifier = svm.SVC(kernel='poly', degree=3, C=C).fit(training_features, training_labels)
    #trained_classifier = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(training_features, training_labels)
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













