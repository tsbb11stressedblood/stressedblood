import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from preprocessing import *

def load_white_cells():
    images = load_images_from_folder('../ground_truth')
    split_cells = [72, 151, 157, 180, 231, 239, 248, 292, 329, 372, 399, 437, 454, 479, 519, 541, 570, 575, 608, 609, 622, 646, 666, 677,
                   795, 818, 827, 971, 1009, 1024, 1028, 1068, 1131, 1144, 1157, 1177, 1192, 1216, 1234, 1287, 1290, 1291, 1343, 1348]
    complicated_cells = [104, 199, 266, 285, 287, 289, 308, 359, 357, 358, 366, 377, 436, 442, 468, 504, 525, 539, 560, 567, 610, 619,
                         653, 656, 680, 729, 737, 774, 776, 786, 803, 815, 829, 845, 868, 918, 1050, 1101, 1132, 1253, 1347, 1431, 1460, 1470, 1474, 1487]
    return split_cells, complicated_cells

def load_test_images():
    images_1 = load_images_from_folder('../nice_areas/12W')
    images_2 = load_images_from_folder('../nice_areas/9W/512x512')
    images_3 = load_images_from_folder('../nice_areas/9W')
    images = images_1 + images_2 + images_3
    return images

def filter_and_segment_image(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_filtered = filter_image(img)
    img_clustered, values = cluster_image(img_filtered)
    foreground, nuclei = separate_image(img_clustered, values)
    nuclei_filtered = remove_edges_from_nuclei(nuclei)
    foreground_filtered = fill_foreground(foreground)
    markers = perform_watershed(foreground_filtered.astype(np.uint8), nuclei_filtered.astype(np.uint8))
    img[markers == -1] = [255,0,0]
    #return img, foreground, foreground_filtered
    return img, nuclei, nuclei_filtered

images = load_test_images()
for img in images:
    segmented_img, nuclei, nuclei_filtered = filter_and_segment_image(img)
    plt.figure('img')
    plt.imshow(segmented_img)
    plt.figure('nuclei')
    plt.imshow(nuclei)
    plt.figure('nuclei_filtered')
    plt.imshow(nuclei_filtered)
    plt.show()
