import numpy as np
import cv2

from matplotlib import pyplot as plt

#padding
p = 15


def preprocessing(image):

    kernel = np.ones((5,5), np.uint8)
    dilate = cv2.dilate(image, kernel, iterations=3)

    blurred = cv2.GaussianBlur(dilate, (5,5), 0)

    return blurred


def extract_cells(orig_image, heat_map_image):
    cell_images_red = []
    cell_images_green = []


    print(orig_image.shape, heat_map_image.shape)

    image = np.uint8(heat_map_image*255.0)


    img_red = preprocessing(heat_map_image[:,:,0])
    img_green = preprocessing(heat_map_image[:,:,1])

    plt.figure('red dilated and blurred heatmap')
    plt.imshow(img_red)
    plt.figure('green dilated and blurred heatmap')
    plt.imshow(img_green)
    #plt.figure('heatmap again?')
    #plt.imshow(image)
    plt.show()

    ret, thresh = cv2.threshold(img_red, 40, 255, 0)

    plt.figure('red thresholded')
    plt.imshow(thresh)
    plt.show()

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im2, contours, -1, (255,255,255), 4)


    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cell_images_red.append(orig_image[y+32-p:y+h+32+p, x+32-p:x+w+32+p])
        #cell_images_red.append(orig_image[x - p:x + w + p, y - p:y + h + p])
        #cv2.rectangle(im2, (x,y), (x+w, y+h), (255,255,255), 2)

    plt.figure('contours for red')
    plt.imshow(im2)
    plt.show()

    #GREEN
    ret, thresh = cv2.threshold(img_green, 200, 255, 0)

    plt.figure('green thresholded')
    plt.imshow(thresh)
    plt.show()

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im2, contours, -1, (255, 255, 255), 4)

    plt.figure('contours for green')
    plt.imshow(im2)
    plt.show()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cell_images_green.append(orig_image[y+32-p:y + h+32+p, x+32-p:x + w+32+p])
        #cell_images_green.append(orig_image[x - p:x + w + p, y - p:y + h + p])


    return cell_images_red, cell_images_green