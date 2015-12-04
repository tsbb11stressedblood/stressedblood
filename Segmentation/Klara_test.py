import numpy as np
import numpy.ma as ma
import cv2
import matplotlib.pyplot as plt
import math
from ColourNM import spacetransformer
from cell import *

import copy

def get_threshold(hist):
    min_ind = np.argmin(hist[15:np.size(hist)/3])
    return min_ind + 15

def mask_joined_WBC(eroded_nuclei_cont, nuclei_mask):
    circles = []
    contours = []
    mask = np.zeros(np.shape(nuclei_mask))
    overlap = np.zeros(np.shape(nuclei_mask))

    for c in eroded_nuclei_cont:
        mom = cv2.moments(c)
        cx = 0
        cy = 0
        if mom['m00'] != 0:
            cx = int(mom['m10']/mom['m00'])
            cy = int(mom['m01']/mom['m00'])

        radius = 1
        area_ratio = 1
        while area_ratio > .9:
            radius += 1
            circ_area = 3.14*radius**2
            nucleus_mask = np.zeros(np.shape(nuclei_mask))
            cv2.circle(nucleus_mask, (cx,cy), radius,(255,255,255),-1)
            nucleus_mask = nucleus_mask * nuclei_mask/255
            area = np.sum(nucleus_mask)
            area_ratio = area/circ_area

        nucleus_mask = np.array(nucleus_mask, dtype=np.uint8)
        circles.append([cx,cy,radius])
        for cir in circles:
             dist = math.sqrt((cir[0]-cx)**2 + (cir[1]-cy)**2)
             if dist <= cir[2] + radius:
                 overlap[nucleus_mask * mask==1] = 1
        mask[nucleus_mask ==1] =1
        _, cont, _ = cv2.findContours(nucleus_mask, 1, 2)
        contours.append(cont)
    return overlap, mask, contours

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
    blur = cv2.GaussianBlur(bg_transform,(5,5),0)
    thres,unknown_mask = cv2.threshold(bg_transform,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Erode and close to remove trash
    unknown_mask = cv2.morphologyEx(unknown_mask,cv2.MORPH_OPEN,kernel, iterations = 2)

    # Mask to get background of the whole cells
    color = 4
    nuclei = spacetransformer.im2c(img, color)
    nuclei = np.array(nuclei * 255, dtype=np.uint8)
    nuclei[unknown_mask == 1] = -1

    # Equalize histogram and blur image
    nuclei = cv2.equalizeHist(nuclei)
    blur = cv2.GaussianBlur(nuclei,(5,5),0)
    blur = cv2.medianBlur(blur,11)

    # Get threshold and mask the image to get nuclei
    hist,_ = np.histogram(blur,255,[0,254])
    thres = get_threshold(hist)
    _,nuclei_mask = cv2.threshold(blur,thres,1,cv2.THRESH_BINARY_INV)
    nuclei_mask[unknown_mask == 1] = 0


    # Get contours and find joined nuclei
    nuclei_circles = nuclei_mask.copy()
    _, all_nuclei_cont, hierarchy = cv2.findContours(nuclei_circles.copy(), 1, 2)
    joined_nuclei_cont = []
    for c in all_nuclei_cont:
        (x,y), radius = cv2.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(nuclei_circles, center, radius, (255,255,255), 1)
        circle_area = 3.14*radius*radius
        nuclei_area = cv2.contourArea(c)
        if nuclei_area/circle_area < .6 and nuclei_area > 150:
            joined_nuclei_cont.append(c)

    # Draw contours of joined nuclei
    empty = np.zeros(np.shape(nuclei))
    empty = np.array(empty*255, dtype=np.uint8)
    joined_nuclei = empty.copy()
    cv2.drawContours(joined_nuclei, joined_nuclei_cont, -1, (255,255,255), -1)


    # Erode joined nuclei and get new contours
    kernelll = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    eroded_joined_nuclei = cv2.erode(joined_nuclei,kernelll,iterations=1)
    _, eroded_nuclei_cont, hierarchy = cv2.findContours(eroded_joined_nuclei.copy(), 1, 2)

    # Get new mask for large nuclei
    overlap, joined_mask, large_nuclei_cont  = mask_joined_WBC(eroded_nuclei_cont, nuclei_mask)
    joined_mask[overlap==1] = 0
    erode_mask = cv2.erode(joined_mask, np.ones(2))
    cells_to_remove = []
    cells_to_remove_im = empty.copy()

    for i,con in enumerate(joined_nuclei_cont):
        temp = empty.copy()
        cv2.drawContours(temp, [con], -1, (255,255,255), -1)
        if np.sum(temp*erode_mask) == 0:
            cells_to_remove_im[temp > 0] = 1

    # Create the markers for the nuclei
    ret, markers_nuc = cv2.connectedComponents(nuclei_mask)
    markers_nuc[markers_nuc == 0] = -1
    markers_nuc = markers_nuc + 1

    # Add the markers for the nuclei and the mask for the whole cells
    markers = markers_nuc | unknown_mask

    # Create a background image to use in watershed
    background =img.copy()
    background[:,:,0] = unknown_mask*255
    background[:,:,1] = unknown_mask*255
    background[:,:,2] = unknown_mask*255

    # Perform watershed and mark the borders on the image
    markers = cv2.watershed(background, markers)
    #img[markers == -1] = [255,0,0]

    # Get an image of the markers and find countours
    cytoplasm_markers = empty.copy()
    cytoplasm_markers[markers == -1] =1
    cytoplasm_markers_im = empty.copy()

    cytoplasm_markers_im[markers > 1] = 1
    cytoplasm_markers_im[markers == -1] = 0
    cytoplasm_markers_im[markers == 1] = 0
    cytoplasm_markers_im = cv2.erode(cytoplasm_markers_im, np.ones(2))

    _,cytoplasm_cont,_ = cv2.findContours(cytoplasm_markers_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    to_be_removed = []
    to_be_exchanged = []
    to_be_inserted = []
    for i,c in enumerate(cytoplasm_cont):
        temp = empty.copy()
        cv2.drawContours(temp, [c], -1, (1,1,1), -1)
        if np.sum(temp*cells_to_remove_im) > 0:
            to_be_removed.append(i)
        if np.sum(temp*joined_mask)> 0:
            to_be_exchanged.append(i)
            for con in large_nuclei_cont:
                joined = empty.copy()
                cv2.drawContours(joined, con, -1, (1,1,1), -1)
                if np.sum(temp*joined) > 0:
                    to_be_inserted.append(con[0])

    for i, cell in enumerate(to_be_exchanged):
        cytoplasm_cont[cell] = to_be_inserted[i]

    for i in sorted(to_be_removed, reverse=True):
        del cytoplasm_cont[i]
    cv2.drawContours(empty, cytoplasm_cont, -1, (255,255,255), -1)
    plt.figure()
    plt.imshow(empty)
    #plt.show()
    return cytoplasm_cont, nuclei_mask

def modify_cell_list(ROI,cytoplasm_cont, nuclei_mask):
    cell_list = []

    print np.size(cytoplasm_cont)
    for c in cytoplasm_cont:
        if cv2.contourArea(c) != 0:
            # Determine the current contour objects area
            area = cv2.contourArea(c)
            # Fit ellipses to cells
            ellipse = cv2.fitEllipse(c)

            # Cut out a boundingbox of the object
            x,y,w,h = cv2.boundingRect(c)
            cell_img = ROI[y:y+h,x:x+w, :]

            # Get the cell mask
            shape = np.shape(cell_img)
            cell_mask = np.zeros((shape[0], shape[1]))
            cv2.drawContours(cell_mask, [c], -1, (1,1,1), -1, offset=(-x,-y))

            # Get the nucleus
            nucleus_mask = nuclei_mask[y:y+h,x:x+w]
            nucleus_mask = nucleus_mask*cell_mask
            area_nuc = np.sum(nucleus_mask)

            cell_list.append(Cell(ellipse, x,y,w,h, area, area_nuc, cell_mask, cell_img))
        else:
            print c
    return cell_list

