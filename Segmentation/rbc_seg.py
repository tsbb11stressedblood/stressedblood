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


# Used for performance measure of the segmentation step
def debug_segmentation(ROI):
    # image init, and conversion to gray and then threshold it
    img = ROI[:, :, 0:3]

    cell_list = []

    # Watershed to find individual cells
    img_fine, ret_fine, markers_fine, markers_nuc = Klara_test.cell_watershed(img)
    cell_list = modify_cell_list(ROI,ret_fine,markers_fine,markers_nuc,cell_list)

    # Class the cell to RBC, background and unknown (possibly WBC!)
    cell_list = RBC_classification(cell_list)
    rbc_counter = rbc_cell_extraction(cell_list)

    # Also extract image
    for cell in cell_list:
        if cell.label == "RBC":
            subroi = ROI[cell.y:cell.y+cell.h, cell.x:cell.x+cell.w, :]
            subroi[cell.mask] = 0
            ROI[cell.y:cell.y+cell.h, cell.x:cell.x+cell.w, :] = subroi

    return rbc_counter, ROI


# Used for debugging/performance measure
def rbc_cell_extraction(cell_list):
    rbc_counter = 0
    for cell in cell_list:
        if cell.label == "RBC":
            rbc_counter += 1
    return rbc_counter


# Segmentation function, call this one with an image or image region to run
# Returns cell_list with all data stored in each cell
def segmentation(ROI):
    # image init, and conversion to gray and then threshold it
    img = ROI[:, :, 0:3]

    cell_list = []

    # Watershed to find individual cells
    #img_fine, ret_fine, markers_fine, markers_nuc, joined_mask, cells_to_remove = Klara_test.cell_watershed(img)
    cytoplasm_cont, nuclei_mask = Klara_test.cell_watershed(img)
    # Put all the objects in the cell_list
    #cell_list = modify_cell_list(ROI,ret_fine,markers_fine,markers_nuc,cell_list)
    cell_list = Klara_test.modify_cell_list(img, cytoplasm_cont, nuclei_mask)
    # Basic classification to RBC and labels these cells
    cell_list = RBC_classification(cell_list)

    # Show the original img
    #fig = plt.figure("klara")
    #ax = fig.add_subplot(111)
    #plt.imshow(img, interpolation='nearest')
    # Print labels
    #print_cell_labels(cell_list, ax)
    #plt.show()

    # Remove all the cells classified to RBC, cell_list now only contains unknown cells
    cell_list = wbc_cell_extraction(cell_list)


    return cell_list

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
    nuclei_mask[nuclei< 0.07*np.amax(nuclei)] = 1 #0.05
    nuclei_mask[nuclei>=0.07*np.amax(nuclei)] = 0

    # Dilate and close to fill the nuclei
    nuclei_mask = cv2.dilate(nuclei_mask, kernel, iterations = 1)
    nuclei_mask = cv2.erode(nuclei_mask, kernel, iterations = 2)
    #nuclei_mask = cv2.morphologyEx(nuclei_mask,cv2.MORPH_CLOSE,kernel, iterations = 2)

    nuclei_mask[nuclei_mask > 0] = 1

    unknown_mask[unknown_mask > 0] = 1

    # Create image with unknown region (between membrane and nucleus)
    unknown_region = nuclei_mask - unknown_mask
    unknown_region[unknown_region > 0] = 1

    #plt.figure("unknown_region")
    #plt.imshow(unknown_region)
    #plt.figure("unknown_mask")
    #plt.imshow(unknown_mask)
    plt.figure("nuclei_mask")
    plt.imshow(nuclei_mask)
    #print(np.amax(unknown_region))
    #print(np.amax(unknown_mask))
    #print(np.amax(nuclei_mask))
    # Create the markers for the nuclei
    ret, markers_nuc = cv2.connectedComponents(nuclei_mask)

    # Add the markers for the nuclei with the mask for the whole cells
    markers = markers_nuc.copy()
    markers += 1
    ret += 1
    markers[unknown_region == 0] = 0

    # Create a background image to use in the watershed
    background = img.copy()
    background[:,:,0] = unknown_mask*255
    background[:,:,1] = unknown_mask*255
    background[:,:,2] = unknown_mask*255


    # Perform watershed and mark the boarders on the image
    markers = cv2.watershed(background, markers)
    img[markers == -1] = [255,0,0]

    return img, ret, markers, nuclei_mask



def modify_cell_list(ROI,ret,markers,markers_nuc_in,cell_list):
    # For each connected object in markers
    markers_nuc = markers_nuc_in.copy()
    markers_nuc[markers_nuc_in >= 1] = 1
    markers_nuc[markers_nuc_in < 1] = 0
    #current_nuc = markers_nuc_in.copy()
    #markers_nuc = np.array(markers_nuc, dtype=np.uint8)

    ellipse_list = []
    for num in range(2,ret):
        img2 = np.array(num==markers, dtype=np.uint8)
        dummy_img, contours, hierarchy = cv2.findContours(img2, 1, 2)
        # Fit ellipses on marked objects in the thresh-image
        # First extract a contour from the list contours
        if len(contours) < 1:
            continue
        if len(contours) > 1:
            index = 0
            area_max = 0
            for ind,i in enumerate(contours):
                # If contours is filled with many objects, the largest object will be chosen
                #  (other objects should be small noise pixels)
                area = cv2.contourArea(i)
                if(area > area_max):
                    area_max = area
                    index = ind

            contour = contours[index]
        else:

            contour = contours[0]
        # Now make sure that this contour is larger than 30 pixels
        if len(contour) < 25:
            continue

        # Fit ellipses to cells
        ellipse = cv2.fitEllipse(contour)
        ellipse_list.append(ellipse)
        # Determine the current contour objects area
        area = cv2.contourArea(contour)

        # Cut out a boundingbox of the object
        x,y,w,h = cv2.boundingRect(contour)
        cell_img = ROI[y:y+h,x:x+w, :]
        cell_mask = markers[y:y+h,x:x+w] == num

        # Determine the current objects nuclei area
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
        point = 0
        # Check if the nucleus is much smaller than the entire cell
        if cell.area_nuc/cell.area < .25:
            point += 2
        # The cell should be in a reasonable size
        if 0.6*RBC_mean_area < cell.area < 3*RBC_mean_area: # upper bound doesnt really matter?
            point += 1
        # Check if the cell is elliptic
        if cell.minor_axis/cell.major_axis < 0.75:
            point += 1
        if point >= 3:
            cell.label = "RBC"
        else:
            cell.label = "U"

    return cell_list

def print_cell_labels(cell_list, ax):
    for cell in cell_list:
        ax.text(cell.x+cell.w/2, cell.y+cell.h/2, cell.label, style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})


def wbc_cell_extraction(cell_list):
    wbc_list = []
    for cell in cell_list:
        if cell.label == "U":
            wbc_list.append(cell)
    return wbc_list

