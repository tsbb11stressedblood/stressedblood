""" Segmentation.py: The initial segmentation of cv2-images. Each Cell will be stored with its cell_img and other data
 in the list cell_list

Author: Marcus Fallqvist and Abdai Ahmed
Date: 2015-10-06
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from ColourNM import spacetransformer
from cell import *
import Klara_test


# Segmentation function, call this one with an image or image region to run
# Returns cell_list with all data stored in each cell
def segmentation(ROI):
    # image init, and conversion to gray and then threshold it
    img = ROI[:, :, 0:3]

    binary_list = []
    cell_list = []

    # Show the original img
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.imshow(img, interpolation='nearest')

    # Watershed to find individual cells
    #img_fine, ret_fine, markers_fine = cell_watershed(img, dist_transform_thresh)
    img_fine, ret_fine, markers_fine, markers_nuc = Klara_test.cell_watershed(img)
    cell_list = modify_cell_list(ROI,ret_fine,markers_fine,markers_nuc,cell_list)
    cell_list = RBC_classification(cell_list)


    # Class the cell to RBC, background and unknown (possibly WBC!)
    cell_list = RBC_classification(cell_list)

    # Print labels
    print_cell_labels(cell_list, ax)
    # Return a list with only WBC
    cell_list = wbc_cell_extraction(cell_list)

    print("Segmentation done")
    plt.show()

    return cell_list

def cell_watershed(img, dist_thresh = 0.7):
    #gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)

    color = 10#10 default 4 och 8 - stronk cellkarna
    gray_transform = spacetransformer.im2c(img, color)
    gray = np.array(gray_transform * 255, dtype=np.uint8)
    grayimg = gray.copy()
    bg_mask = grayimg
    bg_mask[gray>0.5*np.amax(gray)] = 0
    bg_mask[gray<=0.5*np.amax(gray)] = 1
    unknown_mask = gray.copy()
    unknown_mask[gray<=0.5*np.amax(gray)] = 0
    unknown_mask[gray>0.5*np.amax(gray)] = 1

    plt.figure(1000)
    plt.imshow(bg_mask)
    #plt.show()
    plt.figure(1001)
    plt.imshow(unknown_mask)
    kernel = np.ones((3,3),np.uint8)
    color = 8#10 default 4 och 8 - stronk cellkarna
    gray = spacetransformer.im2c(img, color)
    gray = np.array(gray * 255, dtype=np.uint8)
    gray[gray<.35*np.amax(gray)] = -1

    gray = cv2.dilate(gray, kernel, iterations=1)

    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel, iterations = 2)


    #gray = cv2.cvtColor(gray ,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal with a 3x3 kernel

    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    #opening = thresh

    # sure background area, more dilation with more iterations
    sure_bg = bg_mask
    # sure_bg = cv2.dilate(opening,kernel,iterations = 2)

    # Finding sure foreground area, threshold might need changing: lower threshold-factor gives larger sure_fg
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform,dist_thresh*dist_transform.max(),255,0) #0.7 default

    # Finding unknown region, borders of bg-fg
    sure_fg = np.uint8(sure_fg)
    #unknown = cv2.subtract(sure_bg,sure_fg)
    unknown = unknown_mask
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    #plt.figure(15)
    #plt.imshow(markers)
    #plt.show()

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    #print markers.shape, np.amax(markers), np.amin(markers)
    markers = cv2.watershed(img, markers)

    img[markers == -1] = [255,0,0]

    return img, ret, markers


def modify_cell_list(ROI,ret,markers,markers_nuc_in,cell_list):
    # For each connectedobject in markers
    markers_nuc = markers_nuc_in.copy()
    markers_nuc[markers_nuc_in >= 1] = 1
    markers_nuc[markers_nuc_in < 1] = 0
    current_nuc = markers_nuc_in.copy()
    #markers_nuc = np.array(markers_nuc, dtype=np.uint8)
    ellipse_list = []
    for num in range(2,ret):
        img2 = np.array(num==markers, dtype=np.uint8)
        dummy_img, contours, hierarchy = cv2.findContours(img2, 1, 2)

        # Fit ellipses on marked objects in the thresh-image
        # First extract contour from the list contours

        if len(contours)<1:
            continue

        # Now make sure that the contour is larger than 30 pixels
        if len(contours[0]) < 25:
            continue
        # If it is, take the contour and pass it on
        contour = contours[0]

        ellipse = cv2.fitEllipse(contour)
        ellipse_list.append(ellipse)
        # Determine cell area
        area = cv2.contourArea(contour)

        #Cut out region
        x,y,w,h = cv2.boundingRect(contour)
        cell_img = ROI[y:y+h,x:x+w, :]
        cell_mask = markers[y:y+h,x:x+w] == num

        # Nuclei area
        current_nuc = markers_nuc*img2
        img_nuc = np.array(current_nuc, dtype=np.uint8)
        dummy_img, contours_nuc, hierarchy_dummy = cv2.findContours(img_nuc, 1, 2)
        contour_nuc = contours_nuc[0]
        area_nuc = cv2.contourArea(contour_nuc) #markers_nuc[y:y+h, x:x+w]

        # Plots BB
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # Construct cell in the cell_list
        cell_list.append(Cell(ellipse, x,y,w,h, area,area_nuc, cell_mask, cell_img))
    return cell_list

def RBC_classification(cell_list):
    # Get the mean RBC size
    cell_areas = []
    # Extract the median area which most prob. is single RBC area
    for object in cell_list:
        cell_areas.append(object.area)

    RBC_mean_area = np.median(cell_areas)


    # For all cells or bunch of cells, check if they are ellipse-shaped and RBC-size
    for cell in cell_list:
        #print cell.area_nuc/cell.area

        if cell.area_nuc/cell.area < .30: # Cell nucleus is smaller then the cell?
            cell.label = "RBC"
            if 0.6*RBC_mean_area < cell.area < 1.2*RBC_mean_area: # The cell is not to large or small?
                cell.label = "RBC"
                if cell.minor_axis/cell.major_axis < 0.85: # The cell is elliptic?
                    cell.label = "RBC"
        else:
            cell.label = "U"

    return cell_list

def print_cell_labels(cell_list, ax):
    for cell in cell_list:
        if cell.label == "RBC":
            ax.text(cell.x+cell.w/2, cell.y+cell.h/2, 'RBC', style='italic',
            bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
        if cell.label == "Background":
            ax.text(cell.x+cell.w/2, cell.y+cell.h/2, 'BG', style='italic',
            bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
        if cell.label == "U":
            ax.text(cell.x+cell.w/2, cell.y+cell.h/2, 'U', style='italic',
            bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})

def wbc_cell_extraction(cell_list):
    wbc_list = []
    for cell in cell_list:
        if cell.label == "U":
            wbc_list.append(cell)
    return wbc_list

