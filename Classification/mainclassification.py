from classer import *
import cv2
import pylab
from whitecell import *
from classification import *
import math
from Segmentation import rbc_seg
from matplotlib.colors import ListedColormap
import itertools
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics, preprocessing, cross_validation
from sklearn.utils import shuffle
import platform
import pickle # Needed for saving the trainer to file
from PIL import Image

def display_classer(X,y):
    X = reduce_dimension(X,2)
    cmap_light = ListedColormap([[1,0,0,0.5],[0,1,0,0.5],[0,0,1,0.5],[1,1,0,0.5]])
    cmap_bold = ListedColormap([[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,1,0,1]])

    model = training(X, y)

    plt.figure()
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=cmap_bold)

    # Circle out the test data
    #plt.scatter(X[:, 0], X[:, 1], s=80, facecolors='none', zorder=10)

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.axis('off')

    plt.show()


def cross_validation(features, labels):
    '''
    Calculate the cross accuracy for 4 sets from the data
    :param features:
    :param lables:
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


# # FROM white_
# WBC_data = []
# for i in range(1, 101):
#  WBC_array = np.load("white_" + str(i) + ".npy")
#  wc = WhiteCell(WBC_array, -1)
#  wc_features = feature_selection(wc)
#  WBC_data.append(wc_features)
#
# WBC_data= np.asarray(WBC_data)
# np.save("white_feature_array.npy",WBC_data )
#
# # FROM low_en1_
# LOW_EN = []
# for i in range(1, 12):
#     f = Image.open("low_en" + str(i) + ".png")
#     lo_array= np.asarray(f)
#     lo = WhiteCell(lo_array, -1)
#     lo_features = feature_selection(lo)
#     LOW_EN.append(lo_features)
#
# LOW_EN= np.asarray(LOW_EN)
# np.save("low_feature_array.npy",LOW_EN )

"""
#FROM CELL-LIST
RBC_data = []
for i in range(1, 5):
    RBC_array = np.load("others_" + str(i) + ".npy")
    REB_celllist=rbc_seg.segmentation(RBC_array)

    for cell in REB_celllist:
        wc = WhiteCell(cell, -1)
        wc_features = feature_selection(wc)
        RBC_data.append(wc_features)

RBC_data=np.asarray(RBC_data)
print "__________________________________"
#np.save("red_feature_array.npy",RBC_data )
print np.shape(RBC_data)

WBC_data = np.load("white_feature_array.npy")
#RBC_data = np.load("red_feature_array.npy")
print np.shape(WBC_data)

LOW_data = np.load("low_feature_array.npy")
print np.shape(LOW_data)

feature_array = np.append(WBC_data,RBC_data,axis=0)
#feature_array = WBC_data


WBC_labels = np.array([0,1,1,0,1,2,1,0,1,0,0,0,2,2,0,1,1,2,1,1,1,2,2,0,2,1,2,2,1,2,1,2,2,2,2,1,2,2,1,1,2,1,1,1,1,0,2,1,1,1,2,1,1,1,2,2,2,2,0,2,1,1,1,0,2,2,2,2,1,2,2,1,0,1,1,2,2,2,2,2,1,2,1,2,0,1,2,2,1,1,2,0,2,2,0,1,1,1,2,0])
#np.save("WBC_labels.npy",WBC_labels)
#WBC_labels = np.load("WBC_labels.npy")

RBC_labels=3*np.ones([np.shape(RBC_data)[0]])
labels_array = np.append(WBC_labels,RBC_labels,axis=0)
"""

# Run the segmentation on our test image, to get training data.
test_im = cv2.cvtColor(cv2.imread("../npyimages/only_whites_clean.png"), cv2.COLOR_BGR2RGB)
cell_list = rbc_seg.segmentation(test_im)

# Get the labels (visually)
for cell in cell_list[20:30]:
    plt.figure()
    tmp_im = cell.img.copy()
    tmp_im = cv2.drawContours(tmp_im, [cell.contour], -1, (0, 255, 0), 2, offset=(-cell.x, -cell.y))
    plt.imshow(tmp_im)
    plt.title("Size is: " + str(cell.area))
plt.show()

#labels = np.array([])

"""
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
    trainer = pickle.load(f)


total_accuracy = cross_validation(X,y)

#prediction = classify_data(X,trainer)
#confusion = metrics.confusion_matrix(y, prediction)
#print confusion

display_classer(X,y)
"""

