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


# import some data to play with
WBC_data = []
for i in range(1, 21):
    WBC_array = np.load("white_" + str(i) + ".npy")
    wc = WhiteCell(WBC_array, -1)
    wc_features = feature_selection(wc)
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
prediction = cl.classify_data(X,trainer)
confusion = metrics.confusion_matrix(y, prediction)

print confusion

"""
# create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].

Z = trainer.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('SVC with polynomial (degree 3) kernel')
plt.show()

"""