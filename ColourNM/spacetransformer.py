import numpy as np
import cv2


def im2c(im, w2c, color):
    w2c = np.load("w2c.npy")
    color_values = [np.array([0., 0., 0.]), np.array([0., 0., 1.]), np.array([.5, .4, .25]), np.array([.5, .5, .5]),
                    np.array([0., 1., 0.]), np.array([1., .8, 0.]), np.array([1., .5, 1.]), np.array([1., 0., 1.]),
                    np.array([1., 0., 0.]), np.array([1., 1., 1.]), np.array([1., 1., 0.])]

    RR = im[:, :, 0]
    GG = im[:, :, 1]
    BB = im[:, :, 2]

    index_im = 1. + np.floor(RR.flatten(1)/8.0) + 32.0*np.floor(GG.flatten(1)/8.0) + 32.0*32.0*np.floor(BB.flatten(1)/8.0)

    w2cM = w2c[:, color-1]
    tmp = w2cM[index_im.flatten(1).astype(int) - 1]

    rows, cols, chan = im.shape
    out = tmp.reshape(cols, rows)
    out = cv2.transpose(out)
    return out
