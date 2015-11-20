from matplotlib import pyplot as plt
import numpy as np
import cv2
from whitecell import *
from Segmentation import rbc_seg

def feature_selection(whitecell):
    feature_array = []
    feature_array.append(whitecell.get_mean_var_energy(2))
    feature_array.append(whitecell.get_mean_var_energy(7))
    feature_array.append(whitecell.get_mean_var_energy(8))
    feature_array.append(whitecell.hist_features(2,10))
    feature_array.append(whitecell.hist_features(7,10,[0,0.2]))
    feature_array.append(whitecell.hist_features(8,15))
    return [item for sublist in feature_array for item in sublist]
