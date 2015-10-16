""" Segmentation.py: The initial segmentation of cv2-images. Each Cell will be stored with its shapedata in the list cell_list

Author: Marcus Fallqvist and Abdai Ahmed
Date: 2015-10-06
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from cell import *


# Segmentation function, call this one with an image or image region to run
def segmentation(ROI):
    # image load and conversion to gray and then threshold it
    img = ROI[:, :, 0:3]

    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # NOT USING WATERSHED ATM
    # noise removal with a 3x3 kernel
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area, more dilation with more iterations
    sure_bg = cv2.dilate(opening,kernel,iterations=2)

    # Finding sure foreground area, threshold might need changing: lower threshold-factor gives larger sure_fg
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

    # Finding unknown region, borders of bg-fg
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)

    img[markers == -1] = [255,0,0]

    # Show the original img
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img, interpolation='nearest')

    cell_list = []
    ellipse_list = []

    # For each connectedobject in markers
    for num in range(2,ret):
        img2 = np.array(num==markers, dtype=np.uint8)
        dummy_img, contours, hierarchy = cv2.findContours(img2, 1, 2)

        # Fit ellipses on marked objects in the thresh-image
        # First extract contour from the list contours
        if len(contours)<1:
            continue
        # Now make sure that the contour is larger than 30 pixels
        if len(contours[0]) < 30:
            continue
        # If it is, take the contour and pass it on
        contour = contours[0]

        ellipse = cv2.fitEllipse(contour)
        ellipse_list.append(ellipse)
        # Determine cell area
        area = cv2.contourArea(contour)

        #Cut out region
        x,y,w,h = cv2.boundingRect(contour)
        # Plots BB
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # Construct cell in the cell_list
        cell_list.append(Cell( ellipse, x,y,w,h, area, cell_list ))



    # Loop to plot ellipses
    #for _ellipse in ellipse_list:
    # markers = cv2.ellipse(img,_ellipse,(0,255,0),2)
    # Class the cell to RBC, background and unknown (possibly WBC!)
    cell_list = RBC_classification(cell_list)
    print_cell_labels(cell_list, ax)

    # show all binary images
    #plt.figure(8)
    #img3 = np.logical_or(markers==-1,markers==1)
    #plt.imshow(img3)

   # plt.figure(9)
    #plt.imshow(thresh)
    print("SEGMENTATION DONE")
    plt.show()

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
        if cell.minor_axis/cell.major_axis < 0.7:
            if 0.6*RBC_mean_area < cell.area < 1.4*RBC_mean_area:
                cell.label = "RBC"
            else:
                cell.label = "Background"
        else:
            cell.label = "U"

    return cell_list

def print_cell_labels(cell_list, ax):
    for cell in cell_list:
        if cell.label == "RBC":
            ax.text(cell.x+cell.w/2, cell.y+cell.h/2, 'RBC', style='italic',
            bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
        if cell.label == "Background":
            ax.text(cell.x+cell.w/2, cell.y+cell.h/2, 'BG', style='italic',
            bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
        if cell.label == "U":
            ax.text(cell.x+cell.w/2, cell.y+cell.h/2, 'U', style='italic',
            bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})

#'''''''''''''''''''''''''''''''''''''
# Test data, move along!
# imgpath1 = 'smallbloodsmear.jpg'
# imgpath2 = 'test.tif'
# img =  cv2.imread(imgpath2)
# cell_list = segmentation(img)
#print type(img)
#stuff[1][1]
#((97.40660095214844, 176.10752868652344), (24.298473358154297, 43.718692779541016), 73.73028564453125)

# plt.show()
