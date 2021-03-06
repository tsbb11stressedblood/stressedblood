import numpy as np
import cv2

from matplotlib import pyplot as plt

#padding
p = 15


def preprocessing(image):

    #kernel = np.ones((5,5), np.uint8)
    kernel = np.ones((5, 5))
    dilate = cv2.dilate(image, kernel, iterations=3)

    blurred = cv2.GaussianBlur(dilate, (5,5), 0)

    return blurred


def extract_cells(orig_image, heat_map_image):
    cell_images_red = []
    cell_images_green = []

    cell_images_green_confidence = []
    cell_images_red_confidence = []

    print(orig_image.shape, heat_map_image.shape)

    #image = np.uint8(heat_map_image*255.0)


    print "extract cells heat map image shape:", heat_map_image.shape
    img_red = preprocessing(heat_map_image[:,:,0])
    img_green = preprocessing(heat_map_image[:,:,1])


    #tmp = img_green.astype(np.float)

    #filter things
    img_green = ((img_green.astype(np.float)-img_red.astype(np.float)*2)>0)*img_green
    print ("dtype:", img_green.dtype)

    plt.figure('red dilated and blurred heatmap')
    plt.imshow(img_red)
    #cv2.imwrite('red_dilated_and_blurred_heatmap.png', img_red)
    plt.imsave(arr=img_red, fname='red_dilated_and_blurred_heatmap.png')

    fig = plt.figure('green dilated and blurred heatmap')
    #cv2.imwrite('green_dilated_and_blurred_heatmap.png', img_green)
    plt.imshow(img_green)
    #fig.savefig('green_dilated_and_blurred_heatmap.png', bbox_inches='tight', pad_inches=0)
    plt.imsave(arr=img_green, fname='green_dilated_and_blurred_heatmap.png')

    fig = plt.figure('ROI Image')
    plt.imshow(orig_image)
    #cv2.imwrite('ROI_image.png', orig_image)
    #fig.savefig('ROI_image.png', bbox_inches='tight', pad_inches=0)
    plt.imsave(arr=orig_image, fname='ROI_image.png')
    #plt.figure('heatmap again?')
    #plt.imshow(image)
    plt.show()

    print "imgred: ", img_red.shape

    ret, thresh = cv2.threshold(img_red.astype(np.uint8), 40, 255, 0)

    plt.figure('red thresholded')
    plt.imshow(thresh)
    plt.show()
    plt.imsave(arr=thresh, fname='red_thresholded.png')

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im2, contours, -1, (255,255,255), 4)


    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        #im = orig_image[y+32-p:y+h+32+p, x+32-p:x+w+32+p]
        #imgr = img_red[y + 32 - p:y + h + 32 + p, x + 32 - p:x + w + 32 + p]
        im = orig_image[y-p:y+h+p, x-p:x+w+p]
        imgr = img_red[y - p:y + h + p, x - p:x + w  + p]
        if im.shape[0] * im.shape[1] > 400:
            cell_images_red.append(im)
            #im[im == 0] = np.nan
            #cell_images_red_confidence.append(np.sum(im)/(im.shape[0] * im.shape[1]) )
            cell_images_red_confidence.append(np.mean(imgr[imgr>0]))
            #cell_images_red.append(orig_image[x - p:x + w + p, y - p:y + h + p])
            #cv2.rectangle(im2, (x,y), (x+w, y+h), (255,255,255), 2)

    fig = plt.figure('contours for red')
    plt.imshow(im2)
    #cv2.imwrite('contours_for_red.png', im2)
    #fig.savefig('contours_for_red.png', bbox_inches='tight', pad_inches=0)
    plt.imsave(arr=im2, fname='contours_for_red.png')

    plt.show()

    #GREEN
    ret, thresh = cv2.threshold(img_green.astype(np.uint8), 200, 255, 0)

    plt.figure('green thresholded')
    plt.imshow(thresh)
    plt.show()
    plt.imsave(arr=thresh, fname='green_thresholded.png')

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im2, contours, -1, (255, 255, 255), 4)

    fig = plt.figure('contours for green')
    plt.imshow(im2)
    #cv2.imwrite('contours_for_green.png',im2)
    #fig.savefig('contours_for_green.png', bbox_inches='tight', pad_inches=0)
    plt.imsave(arr=im2, fname='contours_for_green.png')
    plt.show()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #im = orig_image[y+32-p:y + h+32+p, x+32-p:x + w+32+p]
        #imgg = img_green[y+32-p:y + h+32+p, x+32-p:x + w+32+p]
        im = orig_image[y-p:y + h+p, x-p:x + w+p]
        imgg = img_green[y-p:y + h+p, x-p:x + w+p]
        if im.shape[0]*im.shape[1] > 800:
            cell_images_green.append(im)
            #im[im==0] = np.nan
            #cell_images_green_confidence.append(np.sum(im) / (im.shape[0] * im.shape[1]))
            cell_images_green_confidence.append(np.mean(imgg[imgg>0]))
            #cell_images_green.append(orig_image[x - p:x + w + p, y - p:y + h + p])


    return cell_images_red, cell_images_red_confidence, cell_images_green, cell_images_green_confidence