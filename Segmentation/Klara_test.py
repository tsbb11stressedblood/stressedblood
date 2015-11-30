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

    # Mask the background to get the whole cells
    color = 10
    bg_transform = spacetransformer.im2c(img, color)
    bg_transform = np.array(bg_transform * 255, dtype=np.uint8)
    bg_img = bg_transform.copy()
    unknown_mask = bg_transform.copy()
    unknown_mask[bg_img<=0.45*np.amax(bg_img)] = 0
    unknown_mask[bg_img>0.45*np.amax(bg_img)] = 1

    # Erode and close to remove trash
    #unknown_mask = cv2.erode(unknown_mask, kernel, iterations =1)
    unknown_mask = cv2.morphologyEx(unknown_mask,cv2.MORPH_CLOSE,kernel, iterations = 2)

    # Mask to get the nuclei
    color = 4
    nuclei = spacetransformer.im2c(img, color)
    nuclei = np.array(nuclei * 255, dtype=np.uint8)
    nuclei_mask = nuclei.copy()
    nuclei_mask[nuclei< 0.05*np.amax(nuclei)] = 1
    nuclei_mask[nuclei>=0.05*np.amax(nuclei)] = 0

    # Dilate and close to fill the nuclei
    nuclei_mask = cv2.erode(nuclei_mask, kernel, iterations =1)
    nuclei_mask = cv2.morphologyEx(nuclei_mask,cv2.MORPH_CLOSE,kernel, iterations = 2)
    nuclei_mask[nuclei_mask > 0] = 1

    # Create image with unknown region (between membrane and nucleus)
    unknown_region = unknown_mask - nuclei_mask

    # Create the markers for the nuclei
    ret, markers_nuc = cv2.connectedComponents(nuclei_mask)

    # Add the markers for the nuclei with the mask for the whole cells
    markers = markers_nuc.copy()
    markers += 1
    ret += 1
    markers[unknown_region == 0] = 0

    # Create a background image to use in the watershed
    background =img.copy()
    background[:,:,0] = unknown_mask*255
    background[:,:,1] = unknown_mask*255
    background[:,:,2] = unknown_mask*255


    # Perform watershed and mark the boarders on the image
    markers = cv2.watershed(background, markers)
    img[markers == -1] = [255,0,0]

    return img, ret, markers, nuclei_mask

#cell_watershed(np.load("simple_test.npy"))
