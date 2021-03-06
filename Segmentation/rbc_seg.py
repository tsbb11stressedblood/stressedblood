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

import pickle
# Used for performance measure of the segmentation step
def debug_segmentation(ROI):
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

    rbc_counter = rbc_cell_extraction(cell_list)

    # Also extract image
    for cell in cell_list:
        if cell.label == "RBC":
            subroi = ROI[cell.y:cell.y+cell.h, cell.x:cell.x+cell.w, :]
            subroi[:, :, 0] = subroi[:, :, 0]*np.invert(cell.mask.astype(bool))
            subroi[:, :, 1] = subroi[:, :, 1]*np.invert(cell.mask.astype(bool))
            subroi[:, :, 2] = subroi[:, :, 2]*np.invert(cell.mask.astype(bool))
            ROI[cell.y:cell.y+cell.h, cell.x:cell.x+cell.w, :] = subroi
            #ROI[cell.mask.astype(int)] = 0

    return rbc_counter, ROI


# Used for debugging/performance measure
def rbc_cell_extraction(cell_list):
    rbc_counter = 0
    for cell in cell_list:
        if cell.label == "RBC":
            rbc_counter += 1
    return rbc_counter


def smooth_contours(cell_list):
    # Smooth the contours
    for i,cell in enumerate(cell_list):
        new_mask = cv2.erode(cell.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),iterations=2 )
        new_mask = cv2.dilate(new_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),iterations=2 )
        """
        print len(cell_list)
        if i < 30:
            plt.figure(4)
            plt.subplot(6,5,i+1)
            plt.imshow(new_mask)
        elif i < 60:
            plt.figure(5)
            plt.subplot(6,5,i-30+1)
            plt.imshow(new_mask)
        #else:
        #    plt.figure(6)
        #    plt.subplot(6,5,i-60+1)
        #    plt.imshow(new_mask)

        if i < 30:
            plt.figure(1)
            plt.subplot(6,5,i+1)
            plt.imshow(cell.mask)
        elif i < 60:
            plt.figure(2)
            plt.subplot(6,5,i-30+1)
            plt.imshow(cell.mask)
        #else:
        #    plt.figure(3)
        #    plt.subplot(6,5,i-60+1)
        #    plt.imshow(cell.mask)

        """
        if np.sum(new_mask) > 0:
            cell.mask = new_mask
            cell.area = np.sum(new_mask>0)
            if cell.area_nuc > cell.area:
                cell.area_nuc = cell.area
    #plt.show()
    return cell_list

# This segmentation variant is used if one needs to remove the segmented RBCs from the ROI aswell
def segment_and_remove_from_roi(ROI):
    # First run the usual segmentation algor
    # image init, and conversion to gray and then threshold it
    img = np.copy(ROI[:, :, 0:3])

    # Watershed to find individual cells
    #img_fine, ret_fine, markers_fine, markers_nuc, joined_mask, cells_to_remove = Klara_test.cell_watershed(img)
    cytoplasm_cont, nuclei_mask, removed_cells, exchanged_cells = Klara_test.cell_watershed(img)
    # Put all the objects in the cell_list
    #cell_list = modify_cell_list(ROI,ret_fine,markers_fine,markers_nuc,cell_list)
    cell_list = Klara_test.modify_cell_list(img, cytoplasm_cont, nuclei_mask)
    # Basic classification to RBC and labels these cells
    cell_list = RBC_classification(cell_list)

    # Smooth the contours
    cell_list = smooth_contours(cell_list)
    
    # Now loop through and remove the segmented rois from the ROI
    for cell in cell_list:
        if cell.label == "RBC":
            subroi = img[cell.y:cell.y+cell.h, cell.x:cell.x+cell.w, :]
            subroi0 = subroi[:, :, 0]
            subroi1 = subroi[:, :, 1]
            subroi2 = subroi[:, :, 2]

            subroi0 = subroi0*np.invert(cell.mask.astype(bool))
            subroi1 = subroi1*np.invert(cell.mask.astype(bool))
            subroi2 = subroi2*np.invert(cell.mask.astype(bool))

            subroi[:, :, 0] = subroi0
            subroi[:, :, 1] = subroi1
            subroi[:, :, 2] = subroi2

            img[cell.y:cell.y+cell.h, cell.x:cell.x+cell.w, :] = subroi

    # Remove all the cells classified to RBC, cell_list now only contains unknown cells
    cell_list = wbc_cell_extraction(cell_list)

    # Before returning, make sure that the image is a bigger bounding box (for viewing purposes)
    for cell in cell_list:
        # First add center coordinates for the bounding box
        xf = cell.x + float(cell.w)/2.
        yf = cell.y + float(cell.h)/2.

        # Get the size of the ROI
        roi_rows = ROI.shape[0]
        roi_cols = ROI.shape[1]

        # Check so that the image is not outside of the ROI
        if (yf - 50 < 0) or (yf + 50 > roi_rows) or (xf - 50 < 0) or (xf + 50 > roi_cols):
            # If it is, then simply use what we already have.. IDGAF
            cell.big_img = ROI[cell.y:cell.y+cell.h, cell.x:cell.x+cell.w, :]
        else:
            cell.big_img = ROI[int(yf - 50):int(yf + 50), int(xf - 50):int(xf + 50), :]

    return cell_list, img


# Segmentation function, call this one with an image or image region to run
# Returns cell_list with all data stored in each cell
def segmentation(ROI):
    # image init, and conversion to gray and then threshold it
    img = ROI[:, :, 0:3]

    cell_list = []

    # Watershed to find individual cells
    #img_fine, ret_fine, markers_fine, markers_nuc, joined_mask, cells_to_remove = Klara_test.cell_watershed(img)
    cytoplasm_cont, nuclei_mask, removed_cells, exchanged_cells = Klara_test.cell_watershed(img)

    # Put all the objects in the cell_list
    #cell_list = modify_cell_list(ROI,ret_fine,markers_fine,markers_nuc,cell_list)
    cell_list = Klara_test.modify_cell_list(img, cytoplasm_cont, nuclei_mask)

    #fff = open('../others_1.pik', 'w+')
    #pickle.dump(cell_list, fff)

    # Basic classification to RBC and labels these cells
    cell_list = RBC_classification(cell_list)

    # Remove all the cells classified to RBC, cell_list now only contains unknown cells
    cell_list = wbc_cell_extraction(cell_list)
    #smooth contours to remove tentacles
    cell_list = smooth_contours(cell_list)

    # Before returning, make sure that the image is a bigger bounding box (for viewing purposes)
    for cell in cell_list:
        # First add center coordinates for the bounding box
        xf = cell.x + float(cell.w)/2.
        yf = cell.y + float(cell.h)/2.

        # Get the size of the ROI
        roi_rows = ROI.shape[0]
        roi_cols = ROI.shape[1]

        # Check so that the image is not outside of the ROI
        if (yf - 50 < 0) or (yf + 50 > roi_rows) or (xf - 50 < 0) or (xf + 50 > roi_cols):
            # If it is, then simply use what we already have.. IDGAF
            cell.big_img = cell.img
        else:
            cell.big_img = ROI[int(yf - 50):int(yf + 50), int(xf - 50):int(xf + 50), :]

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

