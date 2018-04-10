<<<<<<< HEAD
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from kappa import kappa

def make_ellipse_dist_map(I, center, axes):
    m, n = np.shape(I)
    psi = np.zeros([m, n])
    cv2.ellipse(psi, (m/2+center[0], n/2+center[1]), axes, 0, 0, 360, 1, -1)
    psi_inv = -((psi == 0) * (-1) + 1) + 1
    psi_dist = -cv2.distanceTransform(psi.astype(np.uint8), cv2.DIST_L2, 5)
    psi_inv_dist = cv2.distanceTransform(psi_inv.astype(np.uint8), cv2.DIST_L2, 5)

    psi_dist = psi_dist + psi_inv_dist
    return psi, psi_dist


def dirac(t):
    return 1/(math.pi * (1 + t*t))

def Heaviside(t):
    return (1/2.0)*(1+ (2/math.pi)*np.arctan(t))

def chanvese(I, mask, num_iter, mu):
    P = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    #m,n = np.shape(P)
    #psi = np.zeros([m, n])
    #cv2.ellipse(psi, (m/2, n/2), (25, 15), 0, 0, 360, 1, -1)
    #plt.figure()
    #plt.imshow(psi)
    #plt.show()
    #psi_inv = -((psi == 0) * (-1) + 1) + 1
    #psi_dist = -cv2.distanceTransform(psi.astype(np.uint8), cv2.DIST_L2, 5)
    #psi_inv_dist = cv2.distanceTransform(psi_inv.astype(np.uint8), cv2.DIST_L2, 5)

    #psi_dist = psi_dist + psi_inv_dist
    psi0, psi0_dist = make_ellipse_dist_map(P, (0, 0), (25, 10))
    psi1, psi1_dist = make_ellipse_dist_map(P, (-20,-20), (20,25))

    img_inv = -((mask==0) * (-1) + 1) + 1
    #img_inv = -mask
    plt.figure('psi0 dist')
    plt.imshow(psi0_dist)
    plt.figure('psi1 dist')
    plt.imshow(psi1_dist)

    plt.figure('psi0 dist>0')
    plt.imshow(psi0_dist>0)
    plt.figure('psi1 dist>0')
    plt.imshow(psi1_dist>0)
    plt.show()

    #im2, contours, hierarchy = cv2.findContours(psi0.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img2 = np.zeros_like(img)

    #img_contour = cv2.drawContours(img2, contours, -1, 255, 1)
    #img2 = cv2.distanceTransform(img2.astype(np.uint8) - 255, cv2.DIST_L2, 5)
    img2 = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    img3 = cv2.distanceTransform(img_inv.astype(np.uint8), cv2.DIST_L2, 5)
    phi0 = (-img2 + img3*img_inv)
    phi1 = (-img2 + img3*img_inv)
    #plt.ion()
    #fig = plt.figure('phi0')
    #ax = fig.add_subplot(111)
    #plotimg = plt.imshow(phi0)
    #plt.show()


    for i in range(num_iter):
        inidx0, _ = np.where(phi0 >= 0)
        outidx0, _ = np.where(phi0 < 0)

        inidx1, _ = np.where(phi1 >= 0)
        outidx1, _ = np.where(phi1 < 0)

        f = P.copy()

        Hx1vx2 = Heaviside(psi0) + Heaviside(psi1) - Heaviside(psi0)*Heaviside(psi1)
        #plt.figure()
        #plt.clf()

        #plt.show()

        #c10 = sum(sum(L0 * Hx1vx2)) / (inidx0.shape[0])
        uin = sum(sum(f * Hx1vx2)) / sum(sum(Hx1vx2))
        #c20 = sum(sum(L0 * (1 - Hx1vx2))) / (outidx0.shape[0])
        uout = sum(sum(f * (1 - Hx1vx2))) / sum(sum(1-Hx1vx2))

        #c11 = sum(sum(L1 * Hx1vx2)) / (inidx1.shape[0])
        #c21 = sum(sum(L1 * (1 - Hx1vx2))) / (outidx1.shape[0])
        #c11 = sum(sum(L1 * Hx1vx2)) / sum(sum(Hx1vx2))
        #c21 = sum(sum(L1 * (1 - Hx1vx2))) / sum(sum(1-Hx1vx2))

        force_image0 = (f - uin)*(f - uin) + (f - uout)*(f - uout)
        force_image1 = (f - uin)*(f - uin) + (f - uout)*(f - uout)

        kappaphi1 = kappa(phi1)
        kappaphi0 = kappa(phi0)

        force1 = dirac(phi1) * (mu * kappaphi1 - force_image1*(1-Heaviside(phi0)))-2*(phi1-psi1)
        force0 = dirac(phi0) * (mu * kappaphi0 - force_image0*(1-Heaviside(phi1)))-2*(phi0-psi0)
        #force = dirac(phi0) * (mu * kappa(phi0) / np.max(np.max(np.abs(kappa(phi0)))) + force_image )
        #force = (mu * kappa(phi0) / np.max(np.max(np.abs(kappa(phi0)))) + force_image )
        #force = force_image
        force0 = force0 / (np.max(np.max(np.abs(force0))))

        force1 = force1 / (np.max(np.max(np.abs(force1))))

        dt = 0.1
        if i%100 == 0:
            print i

        old = phi0
        phi0 = phi0 + dt * force0
        phi1 = phi1 + dt * force1

        #plt.imshow(phi0>0)
        #plt.colorbar()
        #if i % 100 == 0:
        #    plt.show()
    plt.figure('phi0>0')
    plt.imshow(phi0>0)
    plt.figure('phi1>0')
    plt.imshow(phi1>0)
    plt.figure('phi0')
    plt.imshow(phi0)
    plt.figure('phi1')
    plt.imshow(phi1)
    plt.show()

cell_mask = cv2.imread('cellmask.png')
cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
cell_mask = cell_mask
cell_mask_outline, cell_mask_contours, hierarchy = cv2.findContours(cell_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
plt.figure('mask')
plt.imshow(cell_mask>0)

img = cv2.imread('../ground_truth/17green (5).png')
plt.figure('orig img')
plt.imshow(img)
plt.show()
=======
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from kappa import kappa

def make_ellipse_dist_map(I, center, axes):
    m, n = np.shape(I)
    psi = np.zeros([m, n])
    cv2.ellipse(psi, (m/2+center[0], n/2+center[1]), axes, 0, 0, 360, 1, -1)
    psi_inv = -((psi == 0) * (-1) + 1) + 1
    psi_dist = -cv2.distanceTransform(psi.astype(np.uint8), cv2.DIST_L2, 5)
    psi_inv_dist = cv2.distanceTransform(psi_inv.astype(np.uint8), cv2.DIST_L2, 5)

    psi_dist = psi_dist + psi_inv_dist
    return psi, psi_dist


def dirac(t):
    return 1/(math.pi * (1 + t*t))

def Heaviside(t):
    return (1/2.0)*(1+ (2/math.pi)*np.arctan(t))

def chanvese(I, mask, num_iter, mu):
    P = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    #m,n = np.shape(P)
    #psi = np.zeros([m, n])
    #cv2.ellipse(psi, (m/2, n/2), (25, 15), 0, 0, 360, 1, -1)
    #plt.figure()
    #plt.imshow(psi)
    #plt.show()
    #psi_inv = -((psi == 0) * (-1) + 1) + 1
    #psi_dist = -cv2.distanceTransform(psi.astype(np.uint8), cv2.DIST_L2, 5)
    #psi_inv_dist = cv2.distanceTransform(psi_inv.astype(np.uint8), cv2.DIST_L2, 5)

    #psi_dist = psi_dist + psi_inv_dist
    psi0, psi0_dist = make_ellipse_dist_map(P, (0, 0), (25, 10))
    psi1, psi1_dist = make_ellipse_dist_map(P, (-20,-20), (20,25))

    img_inv = -((mask==0) * (-1) + 1) + 1
    #img_inv = -mask
    plt.figure('psi0 dist')
    plt.imshow(psi0_dist)
    plt.figure('psi1 dist')
    plt.imshow(psi1_dist)

    plt.figure('psi0 dist>0')
    plt.imshow(psi0_dist>0)
    plt.figure('psi1 dist>0')
    plt.imshow(psi1_dist>0)
    plt.show()

    #im2, contours, hierarchy = cv2.findContours(psi0.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img2 = np.zeros_like(img)

    #img_contour = cv2.drawContours(img2, contours, -1, 255, 1)
    #img2 = cv2.distanceTransform(img2.astype(np.uint8) - 255, cv2.DIST_L2, 5)
    img2 = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    img3 = cv2.distanceTransform(img_inv.astype(np.uint8), cv2.DIST_L2, 5)
    phi0 = (-img2 + img3*img_inv)
    phi1 = (-img2 + img3*img_inv)
    #plt.ion()
    #fig = plt.figure('phi0')
    #ax = fig.add_subplot(111)
    #plotimg = plt.imshow(phi0)
    #plt.show()


    for i in range(num_iter):
        inidx0, _ = np.where(phi0 >= 0)
        outidx0, _ = np.where(phi0 < 0)

        inidx1, _ = np.where(phi1 >= 0)
        outidx1, _ = np.where(phi1 < 0)

        f = P.copy()

        Hx1vx2 = Heaviside(psi0) + Heaviside(psi1) - Heaviside(psi0)*Heaviside(psi1)
        #plt.figure()
        #plt.clf()

        #plt.show()

        #c10 = sum(sum(L0 * Hx1vx2)) / (inidx0.shape[0])
        uin = sum(sum(f * Hx1vx2)) / sum(sum(Hx1vx2))
        #c20 = sum(sum(L0 * (1 - Hx1vx2))) / (outidx0.shape[0])
        uout = sum(sum(f * (1 - Hx1vx2))) / sum(sum(1-Hx1vx2))

        #c11 = sum(sum(L1 * Hx1vx2)) / (inidx1.shape[0])
        #c21 = sum(sum(L1 * (1 - Hx1vx2))) / (outidx1.shape[0])
        #c11 = sum(sum(L1 * Hx1vx2)) / sum(sum(Hx1vx2))
        #c21 = sum(sum(L1 * (1 - Hx1vx2))) / sum(sum(1-Hx1vx2))

        force_image0 = (f - uin)*(f - uin) + (f - uout)*(f - uout)
        force_image1 = (f - uin)*(f - uin) + (f - uout)*(f - uout)

        kappaphi1 = kappa(phi1)
        kappaphi0 = kappa(phi0)

        force1 = dirac(phi1) * (mu * kappaphi1 - force_image1*(1-Heaviside(phi0)))-2*(phi1-psi1)
        force0 = dirac(phi0) * (mu * kappaphi0 - force_image0*(1-Heaviside(phi1)))-2*(phi0-psi0)
        #force = dirac(phi0) * (mu * kappa(phi0) / np.max(np.max(np.abs(kappa(phi0)))) + force_image )
        #force = (mu * kappa(phi0) / np.max(np.max(np.abs(kappa(phi0)))) + force_image )
        #force = force_image
        force0 = force0 / (np.max(np.max(np.abs(force0))))

        force1 = force1 / (np.max(np.max(np.abs(force1))))

        dt = 0.1
        if i%100 == 0:
            print i

        old = phi0
        phi0 = phi0 + dt * force0
        phi1 = phi1 + dt * force1

        #plt.imshow(phi0>0)
        #plt.colorbar()
        #if i % 100 == 0:
        #    plt.show()
    plt.figure('phi0>0')
    plt.imshow(phi0>0)
    plt.figure('phi1>0')
    plt.imshow(phi1>0)
    plt.figure('phi0')
    plt.imshow(phi0)
    plt.figure('phi1')
    plt.imshow(phi1)
    plt.show()

cell_mask = cv2.imread('cellmask.png')
cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
cell_mask = cell_mask
cell_mask_outline, cell_mask_contours, hierarchy = cv2.findContours(cell_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
plt.figure('mask')
plt.imshow(cell_mask>0)

img = cv2.imread('../ground_truth/17green (5).png')
plt.figure('orig img')
plt.imshow(img)
plt.show()
>>>>>>> 0ea58f3213c1c9043f8e6ace82251a5410c1c428
chanvese(img, (cell_mask>0)*1., 6000, .5*255*255)