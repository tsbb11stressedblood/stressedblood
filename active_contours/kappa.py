<<<<<<< HEAD
import cv2
import numpy as np
from matplotlib import pyplot as plt

def kappa(I):
    ux, uy = np.gradient(I)
    normDu = np.sqrt(ux*ux + uy*uy + 1e-10)

    Nx = ux / normDu
    Ny = uy / normDu
    nxx, _ = np.gradient(Nx)
    _, nyy = np.gradient(Ny)
    k = nxx + nyy

    return k/np.max(np.max(np.abs(k)))
    #I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    #P = np.pad(I, 1, 'constant')
    #m,n = np.shape(I)
    #print m
    #print n
    #fy = P[2:, 1:n+1] - P[1:m+1, 1:n+1]
    #fy = I - np.roll(I, 1, axis=1)
    #fx = P[1:m+1, 2:]-P[1:m+1, 0:n]
    #fx = I - np.roll(I, 1, axis=0)
    #fyy = P[2:, 1:n+1]+P[0:m, 1:n+1]-2 * I
    #fxx = P[1:m+1, 2:]+P[1:m+1, 0:n]-2 * I
    #test1 = P[2:, 2:]-P[0:m, 2:]
    #test2 = P[2:, 0:n]-P[0:m, 0:n]
    #fxy = 0.25 * (test1+test2)

    #fy = P(3:end, 2:n + 1)-P(1:m, 2:n + 1);
    #fx = P(2:m + 1, 3:end)-P(2:m + 1, 1:n);
    #fyy = P(3:end, 2:n + 1)+P(1:m, 2:n + 1)-2 * I;
    #fxx = P(2:m + 1, 3:end)+P(2:m + 1, 1:n)-2 * I;
    #fxy = 0.25. * (P(3:end, 3:end)-P(1:m, 3:end)+P(3:end, 1:n)-P(1:m, 1:n));

    #plt.figure('fy')
    #plt.imshow(fy)
    #plt.figure('fyy')
    #plt.imshow(fyy)
    #plt.show()


    #G = np.sqrt(fx * fx + fy * fy)
    #fxfxfyfy = (fx * fx + fy * fy)
    #K = (fxx * fy * fy - 2 * fxy * fx * fy + fyy * fx * fx) / np.sqrt(fxfxfyfy*fxfxfyfy*fxfxfyfy)
    #KG = K * G;
=======
import cv2
import numpy as np
from matplotlib import pyplot as plt

def kappa(I):
    ux, uy = np.gradient(I)
    normDu = np.sqrt(ux*ux + uy*uy + 1e-10)

    Nx = ux / normDu
    Ny = uy / normDu
    nxx, _ = np.gradient(Nx)
    _, nyy = np.gradient(Ny)
    k = nxx + nyy

    return k/np.max(np.max(np.abs(k)))
    #I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    #P = np.pad(I, 1, 'constant')
    #m,n = np.shape(I)
    #print m
    #print n
    #fy = P[2:, 1:n+1] - P[1:m+1, 1:n+1]
    #fy = I - np.roll(I, 1, axis=1)
    #fx = P[1:m+1, 2:]-P[1:m+1, 0:n]
    #fx = I - np.roll(I, 1, axis=0)
    #fyy = P[2:, 1:n+1]+P[0:m, 1:n+1]-2 * I
    #fxx = P[1:m+1, 2:]+P[1:m+1, 0:n]-2 * I
    #test1 = P[2:, 2:]-P[0:m, 2:]
    #test2 = P[2:, 0:n]-P[0:m, 0:n]
    #fxy = 0.25 * (test1+test2)

    #fy = P(3:end, 2:n + 1)-P(1:m, 2:n + 1);
    #fx = P(2:m + 1, 3:end)-P(2:m + 1, 1:n);
    #fyy = P(3:end, 2:n + 1)+P(1:m, 2:n + 1)-2 * I;
    #fxx = P(2:m + 1, 3:end)+P(2:m + 1, 1:n)-2 * I;
    #fxy = 0.25. * (P(3:end, 3:end)-P(1:m, 3:end)+P(3:end, 1:n)-P(1:m, 1:n));

    #plt.figure('fy')
    #plt.imshow(fy)
    #plt.figure('fyy')
    #plt.imshow(fyy)
    #plt.show()


    #G = np.sqrt(fx * fx + fy * fy)
    #fxfxfyfy = (fx * fx + fy * fy)
    #K = (fxx * fy * fy - 2 * fxy * fx * fy + fyy * fx * fx) / np.sqrt(fxfxfyfy*fxfxfyfy*fxfxfyfy)
    #KG = K * G;
>>>>>>> 0ea58f3213c1c9043f8e6ace82251a5410c1c428
    #return KG