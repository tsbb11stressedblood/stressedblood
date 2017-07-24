import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from kappa import kappa

def Heaviside(t):
    return (1/2.0)*(1+ (2/math.pi)*np.arctan(t))

def chanvese(I, mask, num_iter, mu):
    P = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    m,n = np.shape(P)
    img = np.zeros([m, n])
    cv2.circle(img, (m/2, n/2), 50, (1), -1)
    #img_inv = -((img == 0) * (-2) + 1)

    img_inv = -((mask==0) * (-1) + 1) + 1
    #img_inv = -mask
    #plt.figure('inv')
    #plt.imshow(img_inv)
    #plt.show()

    im2, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img2 = np.zeros_like(img)

    img_contour = cv2.drawContours(img2, contours, -1, 255, 1)
    #img2 = cv2.distanceTransform(img2.astype(np.uint8) - 255, cv2.DIST_L2, 5)
    img2 = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    img3 = cv2.distanceTransform(img_inv.astype(np.uint8), cv2.DIST_L2, 5)
    phi0 = -img2 + img3*img_inv
    plt.figure('phi0')
    plt.imshow(phi0)
    plt.show()

    for i in range(num_iter):
        inidx, _ = np.where(phi0 >= 0)
        outidx, _ = np.where(phi0 < 0)
        L = P.copy()
        c1 = sum(sum(L * Heaviside(phi0))) / (inidx.shape[0])
        c2 = sum(sum(L * (1 - Heaviside(phi0)))) / (outidx.shape[0])
        print c1
        print c2
        force_image = -(L - c1)*(L - c1) + (L - c2)*(L - c2)

        force = mu * kappa(phi0) / np.max(np.max(np.abs(kappa(phi0)))) + force_image
        #force = force_image
        force = force / (np.max(np.max(np.abs(force))))

        dt = 0.5;
        print i
        old = phi0;
        phi0 = phi0 + dt * force;
    plt.figure('phi0')
    plt.imshow(phi0)
    plt.colorbar()
    #plt.show()
    plt.figure('phi0>0')
    plt.imshow(phi0>0)
    plt.show()
cell_mask = cv2.imread('cellmask.png')
cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
cell_mask = cell_mask
cell_mask_outline, cell_mask_contours, hierarchy = cv2.findContours(cell_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#plt.figure('mask')
#plt.imshow(cell_mask>0)
#plt.show()
img = cv2.imread('../ground_truth/17green (5).png')
chanvese(img, (cell_mask>0)*1., 500, 2*255*255)