import numpy as np
from matplotlib import pyplot as plt
import cv2


def read_image(mask, img):
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            if y > 31 and x > 31 and y < img.shape[0] - 32 and x < img.shape[1] - 32 and mask[y][x].any() != 0:
                print(y, x, img.shape)
                newimg = img[y-32:y+32, x-32:x+32, :] #hmm is copy needed?
                print (newimg.shape)
                plt.imshow(newimg*255)
                plt.show()


img = cv2.imread('C:/Users/erikhk/skola/exjobb/stressedblood/images_for_klara/9W/nuclei.png')
read_image(img, img)