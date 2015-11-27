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
import pickle # Needed for saving the trainer to file
import classer


# import some data to play with
WBC_data = []
for i in range(1, 21):
    WBC_array = np.load("white_" + str(i) + ".npy")
    wc = WhiteCell(WBC_array, -1)
    wc_features = classer.feature_selection(wc)
    WBC_data.append(wc_features)

WBC_data= np.asarray(WBC_data)
WBC_labels = np.array([0,1,1,0,1,2,1,0,1,0,0,0,2,2,0,1,1,2,1,1])
#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the first two features. We could
                     # avoid this ugly slicing by using a two-dim dataset
X = WBC_data
#X = cl.reduce_dimension(X, 2)
#y = iris.target
y = WBC_labels

trainer = cl.training(X,y)

with open("trainer.pik", "wb") as f:
    pickle.dump(trainer, f)

with open("trainer.pik", 'rb') as f:
    trainerTest = pickle.load(f)

prediction = cl.classify_data(X,trainerTest)
confusion = metrics.confusion_matrix(y, prediction)

print confusion
