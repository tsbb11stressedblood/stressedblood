from matplotlib import pyplot as plt
import numpy as np
import cv2
import pylab
from whitecell import *
from Segmentation import rbc_seg
import itertools

def feature_selection(whitecell):
    feature_array = []
    feature_array.append(whitecell.label)
    feature_array.append(whitecell.mean)
    feature_array.append(whitecell.var)
    feature_array.append(whitecell.energy)
    hist1 = whitecell.histogram_features(10.0)
    feature_array.append([i for i in hist1])
    #feature_array.append(hist1[1])
    #feature_array.append(hist1[2])
    #feature_array.append(hist1[3])
    hist2 = whitecell.histogram_features(4.0)
    feature_array.append([i for i in hist2])

    return  feature_array
    #return itertools.chain.from_iterable(feature_array)


WBC_array = np.load('simple_test.npy')
list = rbc_seg.segmentation(WBC_array)

wc = WhiteCell(list[0], -1)
print feature_selection(wc)
plt.imshow(list[0].img)
plt.show()
