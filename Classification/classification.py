from matplotlib import pyplot as plt
import numpy as np
import cv2
from whitecell import *
from Segmentation import rbc_seg
import itertools

def feature_selection(whitecell):
    feature_array = []
    #feature_array.append(whitecell.label)
    feature_array.append(whitecell.mean)
    feature_array.append(whitecell.var)
    feature_array.append(whitecell.energy)
    hist1 = whitecell.histogram_features(10.0)
    #feature_array.append(hist1)
    #feature_array.append([i for i in hist1])
    feature_array.append(hist1[1])
    feature_array.append(hist1[2])
    feature_array.append(hist1[3])
    feature_array.append(hist1[0])
    hist2 = whitecell.histogram_features(4.0)
    feature_array.append(hist2[1])
    feature_array.append(hist2[2])
    feature_array.append(hist2[3])
    feature_array.append(hist2[0])
    feature_array.append(whitecell.area_ratio('b', 0.5))
    feature_array.append(whitecell.area_ratio('g', 0.5))
    feature_array.append(whitecell.area_ratio('r', 0.5))
    feature_array.append(whitecell.area_ratio('h', 0.5))
    feature_array.append(whitecell.area_ratio('s', 0.5))
    feature_array.append(whitecell.area_ratio('v', 0.5))
    feature_array.append(whitecell.area_ratio('bw', 0.5))
    feature_array.append(whitecell.area_ratio('b', 0.7))
    feature_array.append(whitecell.area_ratio('g', 0.7))
    feature_array.append(whitecell.area_ratio('r', 0.7))
    feature_array.append(whitecell.area_ratio('h', 0.7))
    feature_array.append(whitecell.area_ratio('s', 0.7))
    feature_array.append(whitecell.area_ratio('v', 0.7))
    feature_array.append(whitecell.area_ratio('bw', 0.7))

    #feature_array.append([i for i in hist2])
    return [item for sublist in feature_array for item in sublist]
    #return itertools.chain.from_iterable(feature_array)
#WBC_array = np.load('white_1.npy')
#list = rbc_seg.segmentation(WBC_array)

#wc = WhiteCell(WBC_array, -1)
#print feature_selection(wc)
#plt.imshow(list[0].img)
#plt.show()
