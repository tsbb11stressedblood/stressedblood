import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import medpy.filter
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def filter_image(img, itr=10, temp=7, search=21):
    #img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_cielab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_l = img_cielab[:,:,0]
    img_filtered = cv2.fastNlMeansDenoising(img_l,None,itr,temp,search)
    return img_filtered

def compare_filters(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_cielab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_l = img_cielab[:,:,0]
    average_gray = cv2.blur(img_gray,(5,5))
    average_l = cv2.blur(img_l,(5,5))
    gaussian_gray = cv2.GaussianBlur(img_gray,(5,5),0)
    gaussian_l = cv2.GaussianBlur(img_l,(5,5),0)
    median_gray = cv2.medianBlur(img_gray,5)
    median_l = cv2.medianBlur(img_l,5)
    bilateral_gray = cv2.bilateralFilter(img_gray,9,75,75)
    bilateral_l = cv2.bilateralFilter(img_l,9,75,75)
    return img_gray, img_l, average_gray, average_l, gaussian_gray, gaussian_l, median_gray, median_l, bilateral_gray, bilateral_l

def cluster_image(img):
    img_reshape = img.reshape((img.shape[0] * img.shape[1], 1))
    img_reshape = np.float32(img_reshape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(img_reshape,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img.shape))
    return res, center

def separate_image(img, values):
    values = sorted(values)
    mask = np.zeros((img.shape[0], img.shape[1]))
    foreground_value = values[1]
    nuclei_value = values[0]
    foreground = np.zeros((img.shape[0], img.shape[1]))
    nuclei = np.zeros((img.shape[0], img.shape[1]))
    foreground[img <= foreground_value] = 1
    nuclei[img <= nuclei_value] = 1
    return foreground, nuclei

def remove_edges_from_nuclei(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    img = cv2.erode(img,kernel,iterations = 1)
    return img

def fill_foreground(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations = 3)
    return img

def perform_watershed(foreground, nuclei):
    ret, markers = cv2.connectedComponents(nuclei)
    foreground = 1 - foreground
    markers = foreground + markers
    img_test = np.zeros((foreground.shape[0], foreground.shape[1], 3), dtype=np.uint8)
    img_test[:,:,0] = 1-foreground
    img_test[:,:,1] = 1-foreground
    img_test[:,:,2] = 1-foreground
    markers_watershed = cv2.watershed(img_test,markers)
    return markers_watershed

#img = cv2.imread('../nice_areas/9W/512x512/2015-10-15 18.06_1.png')
#img = cv2.imread('../nice_areas/9W/512x512/2015-10-15 18.17_1.png')
#img = cv2.imread('../nice_areas/9W/512x512/2015-10-13 13.48_2.png')
#img = cv2.imread('../nice_areas/12W/2016-01-23 12.07_2.png')
#img = cv2.imread('../nice_areas/12W/2016-01-24 00.45_1.png')
#img = cv2.imread('../nice_areas/12W/2016-01-23 12.07_1.png')
#img = cv2.imread('../nice_areas/9W/2015-10-15 18.17_1.png')
#img = cv2.imread('../nice_areas/9W/2015-10-15 18.06_1.png')
#img = cv2.imread('../nice_areas/9W/2015-10-15 18.06_2.png')
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

images = load_images_from_folder('../ground_truth')
split_cells =[72, 151, 157, 180, 231, 239, 248, 292, 329, 372, 399, 437, 454, 479, 519, 541, 570, 575, 608, 609, 622, 646, 666, 677,
              795, 818, 827, 971, 1009, 1024, 1028, 1068, 1131, 1144, 1157, 1177, 1192, 1216, 1234, 1287, 1290, 1291, 1343, 1348]
complicated_cells = [104, 199, 266, 285, 287, 289, 308, 359, 357, 358, 366, 377, 436, 442, 468, 504, 525, 539, 560, 567, 610, 619,
                     653, 656, 680, 729, 737, 774, 776, 786, 803, 815, 829, 845, 868, 918, 1050, 1101, 1132, 1253, 1347, 1431, 1460, 1470, 1474, 1487]
i = 0
#for img in images:
# for nr in split_cells:
#     #print i
#     img = images[nr]
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img_filtered = filter_image(img)
#     img_clustered, values = cluster_image(img_filtered)
#     foreground, nuclei = separate_image(img_clustered, values)
#     nuclei = remove_edges_from_nuclei(nuclei)
#     foreground = fill_foreground(foreground)
#     markers = perform_watershed(foreground.astype(np.uint8), nuclei.astype(np.uint8))
#     img[markers == -1] = [255,0,0]
#
#     plt.figure()
#     plt.imshow(img_filtered)
#     plt.figure()
#     plt.imshow(img_clustered)
#     plt.figure()
#     plt.imshow(img)
#     plt.show()
#     #i = i+1

# for nr in split_cells:
#     #print i
#     img = images[nr]
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     img_cielab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     img_l = img_cielab[:,:,0]
#     #average_gray = cv2.blur(img_gray,(5,5))
#     average_l = cv2.blur(img,(5,5))
#     average_l = cv2.blur(average_l,(5,5))
#     average_l = cv2.blur(average_l,(5,5))
#     average_l = cv2.cvtColor(average_l, cv2.COLOR_RGB2GRAY)
#
#     #gaussian_gray = cv2.GaussianBlur(img_gray,(5,5),0)
#     gaussian_l = cv2.GaussianBlur(img,(5,5),0)
#     gaussian_l = cv2.GaussianBlur(gaussian_l,(5,5),0)
#     gaussian_l = cv2.GaussianBlur(gaussian_l,(5,5),0)
#     gaussian_l = cv2.cvtColor(gaussian_l, cv2.COLOR_RGB2GRAY)
#
#     #median_gray = cv2.medianBlur(img_gray,5)
#     median_l = cv2.medianBlur(img_l,5)
#     #bilateral_gray = cv2.bilateralFilter(img_gray,9,75,75)
#     bilateral_l = cv2.bilateralFilter(img_l,9,75,75)
#
#     plt.figure('img')
#     plt.imshow(img)
#     plt.figure('gray')
#     plt.imshow(img_gray)
#     plt.figure('l')
#     plt.imshow(img_l)
#     #plt.figure('average_gray')
#     #plt.imshow(average_gray)
#     plt.figure('average_l')
#     plt.imshow(average_l)
#     #plt.figure('gaussian_gray')
#     #plt.imshow(gaussian_gray)
#     plt.figure('gaussian_l')
#     plt.imshow(gaussian_l)
#     #plt.figure('median_gray')
#     #plt.imshow(median_gray)
#     plt.figure('meadian_l')
#     plt.imshow(median_l)
#     #plt.figure('bilateral_gray')
#     #plt.imshow(bilateral_gray)
#     plt.figure('bilateral_l')
#     plt.imshow(bilateral_l)
#
#     plt.show()
#     #i = i+1


# for nr in split_cells:
#     #print i
#     img = images[nr]
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     img_cielab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     img_l = img_cielab[:,:,0]
#
#     img_filtered = filter_image(img)
#     img_clustered, values = cluster_image(img_filtered)
#     foreground, nuclei = separate_image(img_clustered, values)
#     foreground_2 = fill_foreground(foreground)
#     print foreground.dtype
#     foreground_3 = foreground.astype(np.uint8)
#     #des = cv2.bitwise_not(foreground)
#     _,contour,hier = cv2.findContours(foreground_3,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contour:
#         cv2.drawContours(foreground_3,[cnt],0,255,-1)
#
#     #foreground_3 = cv2.bitwise_not(des)
#
#     plt.figure('img')
#     plt.imshow(img)
#     plt.figure('img_clustered')
#     plt.imshow(img_clustered)
#     plt.figure()
#     plt.imshow(foreground)
#     plt.figure()
#     plt.imshow(foreground_2)
#     plt.figure()
#     plt.imshow(foreground_3)
#     plt.show()
#     #i = i+1


#img = cv2.imread('../nice_areas/9W/512x512/2015-10-15 18.06_1.png')
#img = cv2.imread('../nice_areas/9W/512x512/2015-10-15 18.17_1.png')
#img = cv2.imread('../nice_areas/9W/512x512/2015-10-13 13.48_2.png')
img = cv2.imread('../nice_areas/12W/2016-01-23 12.07_2.png')
#img = cv2.imread('../nice_areas/12W/2016-01-24 00.45_1.png')
#img = cv2.imread('../nice_areas/12W/2016-01-23 12.07_1.png')
#img = cv2.imread('../nice_areas/9W/2015-10-15 18.17_1.png')
#img = cv2.imread('../nice_areas/9W/2015-10-15 18.06_1.png')
#img = cv2.imread('../nice_areas/9W/2015-10-15 18.06_2.png')

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_cielab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img_l = img_cielab[:,:,0]

img_filtered = filter_image(img)
img_clustered, values = cluster_image(img_filtered)
foreground, nuclei = separate_image(img_clustered, values)
foreground_2 = fill_foreground(foreground)
nuclei = remove_edges_from_nuclei(nuclei)
print foreground.dtype
foreground_3 = foreground.astype(np.uint8)
#des = cv2.bitwise_not(foreground)
_,contour,hier = cv2.findContours(foreground_3,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
test = []
for i, cnt in enumerate(contour):
    if cv2.contourArea(cnt) < 2000:
        test.append(cv2.contourArea(cnt))
print np.amax(test)
#hist = plt.hist(test, 2000)

laplacian = cv2.Laplacian(img_filtered,cv2.CV_64F)
problematic_contours = []
for i, cnt  in enumerate(contour):
    if hier[0][i][-1] == -1:
        cv2.drawContours(foreground_3,[cnt],0,255,-1)
    elif hier[0][i][-1] != -1 and cv2.contourArea(cnt) < 400:
        #cv2.drawContours(laplacian, [cnt], 0,255, -1)
        cv2.drawContours(foreground_3,[cnt],0,255,-1)
        if cv2.contourArea(cnt) > 30:
            problematic_contours.append(cnt)
    else:
        cv2.drawContours(foreground_3,[cnt],0,0,-1)
#foreground_3 = cv2.bitwise_not(des)

dist_transform = cv2.distanceTransform(foreground_3,cv2.DIST_L2,5)
dist_transform[nuclei > 0 ] = 50
laplacian[laplacian < 0] = 0
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
value_array = []
for cnt in problematic_contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if x > 5 and y > 5 and y < 2020 and x < 2020:
        empty = np.zeros((h+10,w+10))
        cv2.drawContours(empty, [cnt], 0,255,-1, offset=(-x+5, -y+5))
        laplace_cutout = laplacian[y-5:y+h+5,x-5:x+w+5]
        print empty.shape
        print laplace_cutout.shape
        print laplacian.shape
        print 'x:', x, ' y:', y, ' w:', w, ' h:', h
        area_before = cv2.contourArea(cnt)
        value_before = np.sum(laplace_cutout[empty>0])
        empty_dilate = cv2.dilate(empty, kernel, iterations=3)
        area_after = np.sum(np.sum(empty_dilate))/255
        value_after = np.sum(laplace_cutout[empty_dilate>0])
        value = value_after-value_before
        area = area_after-area_before
        value_per_area = value/area
        #laplace_cutout[empty_dilate > 0] = 255
        #plt.figure()
        #plt.imshow(empty)
        #plt.figure()
        #plt.imshow(laplace_cutout)
        #print value_per_area
        #print 'area after: ', area_after
        #print 'area before: ', area_before
        #print 'value after: ', value_after
        #print 'value before: ', value_before
        #plt.show()
        value_array.append(value_per_area)
hist = plt.hist(value_array, 50)
plt.figure('img')
plt.imshow(img)
plt.figure()
plt.imshow(nuclei)
# plt.figure()
# plt.imshow(foreground)
plt.figure()
plt.imshow(dist_transform)
plt.figure()
plt.imshow(laplacian)
plt.show()