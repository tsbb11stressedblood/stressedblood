import numpy as np
import theano
import lasagne
from cnn import build_cnn
import theano.tensor as T





def get_heatmap(image, stride=8, func=None):

    width = image.shape[1]
    height = image.shape[2]
    part_width = part_height = 64
    heat_map = np.zeros((3, width, height))


    jumpi = (width-part_width)/stride
    jumpj = (height-part_height)/stride

    print width, height, heat_map.shape, jumpi, jumpj

    image_parts = np.zeros((  jumpi*jumpj, 3, part_width, part_height))

    ii = 0
    for i in range(jumpi):
        for j in range(jumpj):
            image_parts[ii, :, :, :] = image[0:3, stride*i:stride*i + part_width, stride*j:stride*j + part_height]
            ii += 1

    res = func(image_parts[:, 0:3, :, :])
    ii = 0
    for r in res:
        #print r
        heat_map[0, stride * (ii % jumpi):stride * (ii % jumpi) + stride, stride * int(ii / jumpj):stride * int(ii / jumpj) + stride] = r[0]
        heat_map[1, stride * (ii % jumpi):stride * (ii % jumpi) + stride, stride * int(ii / jumpj):stride * int(ii / jumpj) + stride] = r[1]
        heat_map[2, stride * (ii % jumpi):stride * (ii % jumpi) + stride, stride * int(ii / jumpj):stride * int(ii / jumpj) + stride] = r[2]
        # print(ii)
        ii += 1

    return np.transpose(heat_map, axes=(1, 2, 0))