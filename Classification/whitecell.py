from matplotlib import pyplot as plt
import numpy as np
import cv2
import copy
from ColourNM import spacetransformer
from Segmentation import cell

class WhiteCell:
    def __init__(self, cell, _label):
        self.img = cell.img
        self.size = float(np.size(self.img))
        #self.make_mask()
        self.mask = cell.mask
        self.nucleus_mask = cell.nucleus_mask
        self.contour = cell.contour
        self.area = cell.area
        self.area_nuc = cell.area_nuc

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
        col[self.mask == 0] = 0
        energy = np.sum(np.power(col, 2))/self.size
        return [np.mean(col), np.var(col), energy]

    def hist_features(self, color, bins, interval = [0,1]):
        col = spacetransformer.im2c(self.img, color)
        col[self.mask == 0] = -1
        hist,_ = np.histogram(col,bins,interval)
        hist = hist/float(np.size(col[col>=0]))
        return hist

    def hist_features_hsv(self, channel, bins, interval = [0,1]):
        col = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV_FULL)
        col = col[:,:,channel]/255.0
        col[self.mask == 0] = -1
        hist, _ = np.histogram(col, bins, interval)
        hist = hist/float(np.size(col[col>=0]))
        return hist