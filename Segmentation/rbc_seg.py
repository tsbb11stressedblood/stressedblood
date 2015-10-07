""" Segmentation.py: The initial segmentation of cv2-images. Each Cell will be stored with its shapedata in the list cell_list

Author: Marcus Fallqvist and Abdai Ahmed
Date: 2015-10-06
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from CellClass import *

# Segmentation function, call this one with an image or image region to run
def segmentation(ROI):
    # image load and conversion to gray and then threshold it
    img = ROI
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    """ # NOT USING WATERSHED ATM
    # noise removal with a 3x3 kernel
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area, more dilation with more iterations
    sure_bg = cv2.dilate(opening,kernel,iterations=4)

    # Finding sure foreground area, threshold might need changing: lower threshold-factor gives larger sure_fg
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
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
    """
    #
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img, interpolation='nearest')
    #print(type(thresh))
    #print(type(markers))

    dummy_img, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cell_list = []
    ellipse_list = []
    for i,cnt in enumerate(contours):
        # Fit ellipses on marked objects in the thresh-image
        if len(cnt)<10:
            continue
        ellipse = cv2.fitEllipse(cnt)
        ellipse_list.append(ellipse)
        # Determine cell area
        area = cv2.contourArea(cnt)
        #Cut out region
        x,y,w,h = cv2.boundingRect(cnt)
        # Plots BB
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # Construct cell in the cell_list
        cell_list.append(Cell( ellipse, x,y,w,h, area, cell_list ))

    #plot ellipses
    #for _ellipse in ellipse_list:
     # markers = cv2.ellipse(img,_ellipse,(0,255,0),2)
    # Class the cell to RBC, background and unknown (possibly WBC!)
    cell_list = RBC_classification(cell_list,ax)
    return cell_list

def RBC_classification(cell_list,ax):
    # Get the mean RBC size
    cell_areas = []
    for object in cell_list:
        cell_areas.append(object.area)

    RBC_mean_area = np.median(cell_areas)
    for cell in cell_list:
        if cell.minor_axis/cell.major_axis < 0.7:
            if 0.6*RBC_mean_area < cell.area < 1.4*RBC_mean_area:
                cell.label = "RBC"
                ax.text(cell.x, cell.y, 'RBC', style='italic',
                bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
            else:
                cell.label = "Background"
        else:
            cell.label = "Unknown"


    return cell_list

# Test data, no need
imgpath1 = 'smallbloodsmear.jpg'
imgpath2 = 'test.tif'
img =  cv2.imread(imgpath2)
cell_list = segmentation(img)
#print type(img)
#stuff[1][1]
#((97.40660095214844, 176.10752868652344), (24.298473358154297, 43.718692779541016), 73.73028564453125)

plt.show()
print("SEGMENTATION DONE")