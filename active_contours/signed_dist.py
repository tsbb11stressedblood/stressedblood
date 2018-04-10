<<<<<<< HEAD
import cv2
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import perform_watershed, load_images_and_name_from_folder, filter_image,separate_image,remove_edges_from_nuclei,fill_foreground
from get_training_data import cluster_image_two
import math

def dirac(t):
    return 1/(math.pi * (1 + t*t))

def H(t):
    return (1/2.0)*(1+ (2/math.pi)*math.atan(t))

img = np.zeros([64,64])

cv2.circle(img, (31,31), 24, (1), -1)

img_inv = -( (img == 0)*(-2) + 1)
plt.figure()
plt.imshow(-img_inv)
plt.show()

im2, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img2 = np.zeros_like(img)

img_contour = cv2.drawContours(img2, contours, -1, 255, 1)
img2 = cv2.distanceTransform(img2.astype(np.uint8) - 255, cv2.DIST_L2, 5)
plt.figure()
plt.imshow(img2*img_inv)
plt.figure()
plt.imshow(img2)
plt.figure()
plt.imshow(img_contour)

plt.show()

img = cv2.imread('../ground_truth/17green (5).png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_filtered = filter_image(img)
res, values = cluster_image_two(img_filtered)
foreground, nuclei = separate_image(img_filtered, values)
nuclei = remove_edges_from_nuclei(nuclei)
foreground = fill_foreground(foreground)
markers_watershed = perform_watershed(foreground.astype(np.uint8), nuclei.astype(np.uint8))
img[markers_watershed == -1] = [255, 0, 0]
cell_mask = np.zeros_like(img)
cell_mask = (markers_watershed==5)*1.0

cv2.imwrite("cellmask.png", cell_mask)

cell_mask_outline, cell_mask_contours, hierarchy = cv2.findContours(cell_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(cell_mask_outline, cell_mask_contours, -1, 255, 1)

masked_img = np.zeros_like(img)

masked_img[:,:,0] = img[:,:,0] * cell_mask
masked_img[:,:,1] = img[:,:,1] * cell_mask
masked_img[:,:,2] = img[:,:,2] * cell_mask

masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)

plt.figure('mask')
plt.imshow(cell_mask)
plt.figure()
plt.imshow(masked_img, cmap='gray')
plt.figure('mask outline')
plt.imshow(cell_mask_outline)
=======
import cv2
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import perform_watershed, load_images_and_name_from_folder, filter_image,separate_image,remove_edges_from_nuclei,fill_foreground
from get_training_data import cluster_image_two
import math

def dirac(t):
    return 1/(math.pi * (1 + t*t))

def H(t):
    return (1/2.0)*(1+ (2/math.pi)*math.atan(t))

img = np.zeros([64,64])

cv2.circle(img, (31,31), 24, (1), -1)

img_inv = -( (img == 0)*(-2) + 1)
plt.figure()
plt.imshow(-img_inv)
plt.show()

im2, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img2 = np.zeros_like(img)

img_contour = cv2.drawContours(img2, contours, -1, 255, 1)
img2 = cv2.distanceTransform(img2.astype(np.uint8) - 255, cv2.DIST_L2, 5)
plt.figure()
plt.imshow(img2*img_inv)
plt.figure()
plt.imshow(img2)
plt.figure()
plt.imshow(img_contour)

plt.show()

img = cv2.imread('../ground_truth/17green (5).png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_filtered = filter_image(img)
res, values = cluster_image_two(img_filtered)
foreground, nuclei = separate_image(img_filtered, values)
nuclei = remove_edges_from_nuclei(nuclei)
foreground = fill_foreground(foreground)
markers_watershed = perform_watershed(foreground.astype(np.uint8), nuclei.astype(np.uint8))
img[markers_watershed == -1] = [255, 0, 0]
cell_mask = np.zeros_like(img)
cell_mask = (markers_watershed==5)*1.0

cv2.imwrite("cellmask.png", cell_mask)

cell_mask_outline, cell_mask_contours, hierarchy = cv2.findContours(cell_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(cell_mask_outline, cell_mask_contours, -1, 255, 1)

masked_img = np.zeros_like(img)

masked_img[:,:,0] = img[:,:,0] * cell_mask
masked_img[:,:,1] = img[:,:,1] * cell_mask
masked_img[:,:,2] = img[:,:,2] * cell_mask

masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)

plt.figure('mask')
plt.imshow(cell_mask)
plt.figure()
plt.imshow(masked_img, cmap='gray')
plt.figure('mask outline')
plt.imshow(cell_mask_outline)
>>>>>>> 0ea58f3213c1c9043f8e6ace82251a5410c1c428
plt.show()