import numpy as np
import cv2
import matplotlib.pyplot as plt
from ColourNM import spacetransformer
import copy

def cell_watershed(img):
    """
    Performs watershed on an RGB-image. Returns the image with the new boarders and the markers
    """
    img = img[:,:,0:3]
    kernel = np.ones((3,3),np.uint8)


    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #plt.figure("HSV")
    #plt.imshow(img_hsv[:,:,2])

    # Mask the background to get the whole cells
    color = 10
    bg_transform = spacetransformer.im2c(img, color)
    bg_transform = np.array(bg_transform * 255, dtype=np.uint8)
    bg_img = bg_transform.copy()
    unknown_mask = bg_transform.copy()
    unknown_mask[bg_img<=0.5*np.amax(bg_img)] = 0
    unknown_mask[bg_img>0.5*np.amax(bg_img)] = 1

    # Erode and close to remove trash
    unknown_mask = cv2.erode(unknown_mask, kernel, iterations =1)
    unknown_mask = cv2.morphologyEx(unknown_mask,cv2.MORPH_CLOSE,kernel, iterations = 2)

    plt.figure("whole cell")
    plt.imshow(unknown_mask)
    # Mask to get the nuclei
    color = 8
    gray = spacetransformer.im2c(img, color)
    gray = np.array(gray * 255, dtype=np.uint8)
    gray[gray<.35*np.amax(gray)] = -1

    #Play with other channels in the future
    #color = 4
    #gray_2 = spacetransformer.im2c(img, color)
    #gray_2 = np.array(gray_2 * 255, dtype=np.uint8)
    #gray_2[gray_2>0.05*np.amax(gray_2)] = -1

    # Dilate and close to fill the nuclei
    gray = cv2.dilate(gray, kernel, iterations=1)
    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel, iterations = 2)
    close[close == 255] = 0
    close[close > 0] = 1

    # Create image with unknown region (between membrane and nucleus)
    unknown_region = unknown_mask - close

    plt.figure("nuclei")
    plt.imshow(close)
    # Create the markers for the nuclei
    ret, markers_nuc = cv2.connectedComponents(close)

    # Add the markers for the nuclei with the mask for the whole cells
    markers_nuc += 1
    ret += 1
    markers_nuc[unknown_region==0] = 0

    # Perform watershed and mark the boarders on the image
    markers = cv2.watershed(img, markers_nuc)
    img[markers == -1] = [255,0,0]

    #plt.figure("res")
    #plt.imshow(img)
    #plt.show()

    return img, ret, markers, close

#cell_watershed(np.load("segmentation_test.npy"))