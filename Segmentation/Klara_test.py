import numpy as np
import numpy.ma as ma
import cv2
import matplotlib.pyplot as plt
import math
from ColourNM import spacetransformer
from cell import *
import cProfile, pstats, StringIO

import copy

def get_threshold(hist):
    ind = 15# np.argmax(hist)
    done = False
    while not done:
        if hist[ind + 1] < hist[ind]:
            ind += 1
        elif hist[ind + 3] < hist[ind]:
            ind += 1
        if hist[ind + 1] > hist[ind] and hist[ind + 3] > hist[ind]:
            done = True
        if ind > 200:
            done = True
    print ind
    return ind

def mask_joined_WBC(eroded_nuclei_cont, nuclei_mask, max_radius):
    """
    Separates the large joined WBC from overlapping objects
    :param eroded_nuclei_cont: List with contours of the eroded joined_nuclei_mask, only large WBC remaining
    :param nuclei_mask: Binary image of the nuclei
    :param max_radius: The largest radius of all enclosing circles for the joined nuclei
    :return:
    """
    circles = []
    shape = np.shape(nuclei_mask)
    image_max_y = shape[0]
    image_max_x = shape[1]
    mask = np.zeros(shape)
    overlap = np.zeros(shape)

    for c in eroded_nuclei_cont:
        mom = cv2.moments(c)
        cx = 0
        cy = 0
        if mom['m00'] != 0:
            cx = int(mom['m10']/mom['m00'])
            cy = int(mom['m01']/mom['m00'])

        y_min = max(cy-max_radius, 0)
        y_max = min(cy + max_radius,image_max_y)
        x_min = max(cx-max_radius, 0)
        x_max = min(cx + max_radius, image_max_x)

        nuclei_mask_temp = nuclei_mask[y_min:y_max,x_min:x_max]
        nucleus_mask = np.zeros(np.shape(nuclei_mask_temp))

        radius = 1
        area_ratio = 1

        while area_ratio > .9:
            radius += 1
            circ_area = 3.14*radius**2
            cv2.circle(nucleus_mask, (int((np.floor((x_max-x_min)/2))), int(np.floor((y_max-y_min)/2))), radius, (1,1,1),-1)
            nucleus_mask = nucleus_mask*nuclei_mask_temp
            area = np.sum(nucleus_mask)
            area_ratio = area/circ_area

        for cir in circles:
            dist = math.sqrt((cir[0]-cx)**2 + (cir[1]-cy)**2)
            if (dist <= cir[2] + radius) & (dist > 0):
                y1 = cy -radius
                x1 = cx -radius
                h1 = radius*2 +1
                y2 = cir[1] -cir[2]
                x2 = cir[0]-cir[2]
                h2 = cir[2]*2+1
                y_min = min(y1, y2)
                y_max = max(y1 + h1,y2 + h2)
                x_min = min(x1,x2)
                x_max = max(x1 + h1, x2 + h2)
                x_tranf = x_max-x_min
                y_tranf = y_max-y_min
                first_circle = np.zeros((y_tranf, x_tranf))
                second_circle = first_circle.copy()
                overlap_circle = first_circle.copy()
                cv2.circle(first_circle, (cx-x_min,cy-y_min), radius,(1,1,1),-1)
                cv2.circle(second_circle, (cir[0]-x_min,cir[1]-y_min), cir[2],(1,1,1),-1)
                overlap_circle[first_circle*second_circle > 0] = 1
                overlap_rect = overlap[y_min:y_max,x_min:x_max]
                overlap_rect[overlap_circle + overlap_rect > 0] = 1
                overlap[y_min:y_max,x_min:x_max] = overlap_rect

        circles.append([cx,cy,radius])
        cv2.circle(mask, (cx,cy), radius,(1,1,1),-1)

    mask[overlap == 1] = 0
    mask[nuclei_mask*mask == 0] = 0
    mask = cv2.erode(mask, np.ones((2,2)))
    mask = np.array(mask, dtype=np.uint8)
    _, cont, _ = cv2.findContours(mask.copy(), 1, 2)
    return mask, cont

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

    # Get colorspace 4 and set the background of whole cells to -1
    color = 4
    nuclei = spacetransformer.im2c(img, color)
    nuclei = np.array(nuclei * 255, dtype=np.uint8)
    nuclei[unknown_mask == 1] = -1

    # Equalize histogram and blur image
    nuclei = cv2.equalizeHist(nuclei)
    blur = cv2.GaussianBlur(nuclei,(5,5),0)
    blur = cv2.medianBlur(blur,11)

    # Get threshold and mask the image to get nuclei set background of the whole cells to 0
    hist,_ = np.histogram(blur,255,[0,254])
    thres = get_threshold(hist)
    _,nuclei_mask = cv2.threshold(blur,thres,1,cv2.THRESH_BINARY_INV)
    nuclei_mask[unknown_mask == 1] = 0

    # Get contours and find joined nuclei, look at hull if possible
    _, all_nuclei_cont, _ = cv2.findContours(nuclei_mask.copy(), 1, 2)
    joined_nuclei_cont = []
    max_radius = 0
    for c in all_nuclei_cont:
        (x,y), radius = cv2.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        if radius > max_radius:
            max_radius = radius
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
    _, eroded_nuclei_cont,_ = cv2.findContours(eroded_joined_nuclei.copy(), 1, 2)


    # Find contours to be removed, assumed to be joined red blood cells
    cells_to_remove = []
    for i,c in enumerate(joined_nuclei_cont):
        x,y,w,h = cv2.boundingRect(c)
        erode = eroded_joined_nuclei[y:y+h,x:x+w]
        temp = np.zeros(np.shape(erode))
        cv2.drawContours(temp, [c], -1, (1,1,1), -1, offset=(-x,-y))
        if np.sum(temp*erode) == 0:
            cells_to_remove.append(c)

    cells_to_remove_im = empty.copy()
    cv2.drawContours(cells_to_remove_im, cells_to_remove, -1, (255,255,255), -1)

    # Get new mask for large nuclei
    joined_mask, large_nuclei_cont = mask_joined_WBC(eroded_nuclei_cont, nuclei_mask, max_radius)

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

    # Get an image of the markers and find contours
    cytoplasm_markers_im = empty.copy()
    cytoplasm_markers_im[markers > 1] = 1
    cytoplasm_markers_im[markers == -1] = 0
    cytoplasm_markers_im = cv2.erode(cytoplasm_markers_im, np.ones(2)) #ska det inte vara en (2,2)
    _,cytoplasm_cont,_ = cv2.findContours(cytoplasm_markers_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    to_be_removed = []
    to_be_inserted = []
    removed_cells = []
    exchanged_cells = []

    for i,c in enumerate(cytoplasm_cont):
        x,y,w,h = cv2.boundingRect(c)
        cell_img = cells_to_remove_im[y:y+h,x:x+w]
        join = joined_mask[y:y+h,x:x+w]
        emp = np.zeros(np.shape(cell_img))
        temp = emp.copy()
        cv2.drawContours(temp, [c], -1, (1,1,1), -1, offset=(-x,-y))
        if np.sum(temp*cell_img) > 0:
            to_be_removed.append(i)
            removed_cells.append(c)
        if np.sum(temp*join)> 0:
            to_be_removed.append(i)
            exchanged_cells.append(c)
            for con in large_nuclei_cont:
                joined = emp.copy()
                cv2.drawContours(joined, [con], -1, (1,1,1), -1,offset=(-x,-y))
                if np.sum(temp*joined) > 0:
                    to_be_inserted.append(con)

    for i in sorted(to_be_removed, reverse=True):
        del cytoplasm_cont[i]

    for con in to_be_inserted:
        cytoplasm_cont.append(con)

    return cytoplasm_cont, nuclei_mask, removed_cells, exchanged_cells

def modify_cell_list(ROI,cytoplasm_cont, nuclei_mask):
    cell_list = []

    for c in cytoplasm_cont:
        if cv2.contourArea(c) > 100:
            # Determine the current contour objects area
            area = cv2.contourArea(c)
            # Fit ellipses to cells
            ellipse = cv2.fitEllipse(c)

            # Cut out a bounding box of the object
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
            cell_list.append(Cell(ellipse, x,y,w,h, area, area_nuc, cell_mask, cell_img, nucleus_mask, c))

    print "Number of cells in cellist: ", cell_list.__len__()
    return cell_list

#im = np.load("../gui/ek_test_2.npy")
#
# im = np.load("../npyimages/testim_3.npy")
# plt.figure()
# plt.imshow(im)
# cytoplasm_cont, nuclei_mask = cell_watershed(im)

#modify_cell_list(im,cytoplasm_cont, nuclei_mask)