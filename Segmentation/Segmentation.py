""" Segmentation.py: The initial segmentation of cv2-images. Each Cell will be stored with its shapedata in the list cell_list

Author: Marcus Fallqvist and Abdai Ahmed
Date: 2015-10-06
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from CellClass import *

# Segmentation function, call this one with an image or image region to run - NOT 100% DONE.
def segmentation( img ):
    # image load and conversion to gray and then threshold it
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #plt.figure(1)
    #plt.imshow(thresh, interpolation='nearest')

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

    plt.figure()
    plt.imshow(img, interpolation='nearest')

    # Find countours and fit ellipses to the cells
    #dum = cv2.cvtColor(markers ,cv2.COLOR_BGR2GRAY)
    #markers[markers == -1] = 1
    print(type(thresh))
    print(type(markers))
    dummyimg, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cell_list = []
    ellipse_list = []
    for i,cnt in enumerate(contours):
        # Fit ellipses on marked objects in the thresh-image
        ellipse = cv2.fitEllipse(cnt)
        ellipse_list.append(ellipse)
        #Cut out region
        x,y,w,h = cv2.boundingRect(cnt)
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #assign that shit: stuff[1][0], stuff[1][1]
        cell_list.append(Cell( ellipse, x,y,w,h ))
        print cell_list[i].label
    #plot that shit
    for _ellipse in ellipse_list:
      markers = cv2.ellipse(img,_ellipse,(0,255,0),2)

    return cell_list


imgpath = 'smallbloodsmear.jpg'
img =  cv2.imread(imgpath)
cell_list = segmentation(img)

#stuff[1][1]

#((97.40660095214844, 176.10752868652344), (24.298473358154297, 43.718692779541016), 73.73028564453125)



plt.show()
print("DONE")