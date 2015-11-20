from matplotlib import pyplot as plt
import numpy as np
import cv2
import copy
from ColourNM import spacetransformer
from Segmentation import cell

class WhiteCell:
    def __init__(self, cell_nparray, _label):
        self.img = cell_nparray
        self.size = float(np.size(self.img))
        self.make_mask()
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