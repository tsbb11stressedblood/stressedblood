from matplotlib import pyplot as plt
import numpy as np
import cv2
from Segmentation import cell

class WhiteCell:
    def __init__(self, cell_nparray, _label):
        self.label =_label
        self.img = cell_nparray
        self.bgr = self.img/255.0
        #self.bgr = cell.img/255.0
        #self.size = float(np.size(cell.img))
        self.size = float(np.size(self.img))
        #self.mask = cell.mask
        #self.hsv = cv2.cvtColor(cell.img, cv2.COLOR_BGR2HSV)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.__convert_to_normalized_hsv()
        self.bw = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)/255.0
        #self.bw = cv2.cvtColor(cell.img, cv2.COLOR_BGR2GRAY)/255.0
        self.mean = []
        self.var = []
        self.energy = []
        self.mean_var()
        self.get_energy()

    def __convert_to_normalized_hsv(self):
        """Normalizes the HSV channel"""
        self.hsv[:,:,0] = self.hsv[:,:,0]/179.0
        self.hsv[:,:,1] = self.hsv[:,:,1]/255.0
        self.hsv[:,:,2] = self.hsv[:,:,2]/255.0

    def mean_var(self):
        """Gets the mean and variance of bgr, hsv and bw. [b g r h s v bw]"""
        for i in range(0,3):
            self.mean.append(np.mean(self.bgr[:,:,i]))
            self.var.append(np.var(self.bgr[:,:,i]))
        for i in range(0,3):
            self.mean.append(np.mean(self.hsv[:,:,i]))
            self.var.append(np.var(self.hsv[:,:,i]))
        self.mean.append(np.mean(self.bw))

    def get_energy(self):
        """Gets the energy of bgr, hsv and bw. [b g r h s v bw]"""
        for i in range(0,3):
            self.energy.append(np.sum(np.power(self.bgr[:,:,i], 2))/self.size)
        for i in range(0,3):
            self.energy.append(np.sum(np.power(self.hsv[:,:,i], 2))/self.size)
        self.energy.append(np.sum(np.power(self.bw, 2))/self.size)

    def histogram(self, channel, bins):
        """Gets the histogram for a chosen channel ('b', 'g', 'r', 'h', 's', 'v', 'bw') and a chosen number of bins"""
        if channel == 'b':
            hist,_ = np.histogram(self.bgr[:,:,0], bins)
        elif channel == 'g':
            hist,_ = np.histogram(self.bgr[:,:,1], bins)
        elif channel == 'r':
            hist,_ = np.histogram(self.bgr[:,:,2], bins)
        elif channel == 'h':
            hist,_ = np.histogram(self.hsv[:,:,0], bins)
        elif channel == 's':
            hist,_ = np.histogram(self.hsv[:,:,1], bins)
        elif channel == 'v':
            hist,_ = np.histogram(self.hsv[:,:,2], bins)
        elif channel == 'bw':
            hist,_ = np.histogram(self.bw, bins)
        return hist

    def histogram_features(self, bins):
        """Calculates the largest bin and the percentage of pixels in that bin.
        Returns an array with the largest bin and the percentages of pixels in that bin for [r, g, b, h, s, v, bw]"""
        largest_bin = []
        perc_in_bin = []
        bin_index = []
        nr_of_bins = []
        dist = []
        for i in range(0,3):
            hist,_ = np.histogram(self.bgr[:,:,i], bins)
            largest_bin.append(np.argmax(hist)/bins)
            perc_in_bin.append(np.amax(hist)/self.size)
            bin_index.append(np.argmax(hist))
            nr_of_bins.append( (hist > np.amax(hist)*0.5).sum()/bins)
        dist.append((np.absolute(bin_index[0]-bin_index[1]) + np.absolute(bin_index[0]-bin_index[2]) + np.absolute(bin_index[1]-bin_index[2]))/bins)
        for i in range(0,3):
            hist,_ = np.histogram(self.hsv[:,:,i], bins)
            largest_bin.append(np.argmax(hist)/bins)
            perc_in_bin.append(np.amax(hist)/self.size)
            nr_of_bins.append( (hist > np.amax(hist)*0.5).sum()/bins)

        hist, _ = np.histogram(self.bw)
        largest_bin.append(np.argmax(hist)/bins)
        perc_in_bin.append(np.amax(hist)/self.size)
        nr_of_bins.append( (hist > np.amax(hist)*0.5).sum()/bins)
        return largest_bin, perc_in_bin, dist, nr_of_bins

    def hist_dist_between_peaks(self, bins):
        """Calculates the distance between the three peaks in the histograms for the b-,g-,r-channel"""
        bin_index=[]
        for i in range(0,3):
            hist,_ = np.histogram(self.bgr[:,:,i], bins)
            bin_index.append(np.argmax(hist))
        dist = (np.absolute(bin_index[0]-bin_index[1]) + np.absolute(bin_index[0]-bin_index[2]) + np.absolute(bin_index[1]-bin_index[2]))/bins
        return dist

    def area_ratio(self, channel, threshold):
        """Calculates the ratio between the cell area and the area for a chosen threshold in a chosen channel"""
        if channel == 'b':
            channel_image = self.bgr[:,:,0]
        elif channel == 'g':
            channel_image = self.bgr[:,:,1]
        elif channel == 'r':
            channel_image = self.bgr[:,:,2]
        elif channel == 'h':
            channel_image= self.hsv[:,:,0]
        elif channel == 's':
            channel_image = self.hsv[:,:,1]
        elif channel == 'v':
            channel_image = self.hsv[:,:,2]
        elif channel == 'bw':
            channel_image = self.hsv[:,:,1]
        channel_image[channel_image>threshold*np.amax(channel_image)] = 1
        channel_image[channel_image<threshold*np.amax(channel_image)] = 0
        area = np.count_nonzero(channel_image)
        ratio = area/self.size
        return [ratio]

    def number_of_nuclei(self):
        """Calculates the number of nuclei"""

