
from __future__ import print_function

import sys
import os
import time
import cv2
import numpy as np
import theano
import theano.tensor as T
import matplotlib.image as mpimg
import fnmatch
import scipy
import lasagne
from PIL import Image
import math
#import lasagne.layers.dnn
from scipy.misc import imresize, imrotate
from cnn import build_cnn
from cnn import load_dataset
from sliding_window import get_heatmap
from extract_cells import extract_cells


import matplotlib.pyplot as plt
from nolearn.lasagne.visualize import *


input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

network = build_cnn(input_var)


prediction = lasagne.layers.get_output(network)

loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

    #get predictions
get_preds = theano.function([input_var], test_prediction, allow_input_downcast=True)

with np.load('model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

#test_image = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.06_2.png', 'r') / np.float32(256.0)
#test_image = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.06_1.png', 'r') / np.float32(256.0)

#test_image = mpimg.imread('../nice_areas/9W/2015-10-15 18.06_1.png', 'r') / np.float32(256.0)


test_image = mpimg.imread('../fake_areas/9W/1.png', 'r') / np.float32(256.0)
#test_image = mpimg.imread('../fake_areas/9W/1.png', 'r')

#test_image = test_image / np.float32(256.0)
#test_image = mpimg.imread('../nice_areas/9W/512x512/2015-10-13 18.02_2.png', 'r') / np.float32(256.0)



print ("means; all, R, G, B: ", test_image.mean(), test_image[:,:,0].mean(), test_image[:,:,1].mean(), test_image[:,:,2].mean())

#test_image[:,:,0] = test_image[:,:,0] - test_image[:,:,0].mean()
#test_image[:,:,1] = test_image[:,:,1] - test_image[:,:,1].mean()
#test_image[:,:,2] = test_image[:,:,2] - test_image[:,:,2].mean()

#test_image = test_image - test_image.mean()

print ("means; all, R, G, B: ", test_image.mean(), test_image[:,:,0].mean(), test_image[:,:,1].mean(), test_image[:,:,2].mean())


heat_map = np.zeros((3,512,512))

    #print ("before:", test_image.shape)
#test_image = np.transpose(test_image, axes=(2, 1, 0))
    #print("after:", test_image.shape)

#images = np.zeros((4096, 3, 64, 64))

#ii = 0

#for i in range(56):
#    for j in range(56):
#        #images[ii,:,:,:] = test_image[0:3, 64*i:64*i+64, 64*j:64*j+64]
#        images[ii, :, :, :] = test_image[0:3, 8 * i:8 * i + 64, 8 * j:8 * j + 64]
#        ii += 1

#testaa = get_preds(images[0:2809, 0:3, :, :])
#testaa = get_preds(images[0:3136, 0:3, :, :])

tt = time.time()
heat_map = get_heatmap(image=np.transpose(test_image, axes=(2, 1, 0)), stride=4, func=get_preds)
print("heat map took: ", time.time()-tt)

#print(np.max(heat_map))

red_cells, red_cells_confidence, green_cells, green_cells_confidence = extract_cells(test_image, heat_map)

print (red_cells_confidence)
print (green_cells_confidence)

for c in red_cells:
    plt.figure()
    plt.imshow(c)
    plt.show()

for c in green_cells:
    plt.figure()
    plt.imshow(c)
    plt.show()


#ii = 0
#for t in testaa:
#    heat_map[0, 8 * (ii % 56):8 * (ii % 56) + 8, 8 * int(ii / 56):8 * int(ii / 56) + 8] = t[0]
#    heat_map[1, 8 * (ii % 56):8 * (ii % 56) + 8, 8 * int(ii / 56):8 * int(ii / 56) + 8] = t[1]
#    heat_map[2, 8 * (ii % 56):8 * (ii % 56) + 8, 8 * int(ii / 56):8 * int(ii / 56) + 8] = t[2]
    #print(ii)
#    ii += 1


#plt.imshow(np.transpose(heat_map, axes=(1, 2, 0)))
plt.imshow(heat_map)
plt.figure()
#plt.imshow(np.transpose(-np.reciprocal(np.log10(heat_map)), axes=(1, 2, 0)))
plt.imshow( -np.reciprocal(np.log10(heat_map)) )
# plt.title("Label: {}".format(testaa[i]))
plt.figure()
plt.imshow(heat_map[:,:,0])
plt.figure()
plt.imshow(heat_map[:,:,1])
plt.show()


#visualize kernels!

# layers = lasagne.layers.get_all_layers(network)
# layercounter = 0
# for l in layers:
#     if 'Conv2DLayer' in str(type(l)):
#         f = open('layer' + str(layercounter) + '.weights', 'wb')
#         weights = l.W.get_value()
#         weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2], weights.shape[3])
#         # weights[0]
#         for i in range(weights.shape[0]):
#             wmin = float(weights[i].min())
#             wmax = float(weights[i].max())
#             weights[i] *= (255.0 / float(wmax - wmin))
#             weights[i] += abs(wmin) * (255.0 / float(wmax - wmin))
#         np.save(f, weights)
#         f.close()
#         layercounter += 1
#
#
# with open('layer0.weights', 'rb') as f:
#     layer0 = np.load(f)
#
# fig, ax = plt.subplots(nrows=3, ncols=32, sharex=True, sharey=False)
# #sorg = fig.add_subplot(3,32,1)
# for i in xrange(1,97):
# #s = fig.add_subplot(3,32,i)
# #s.set_adjustable('box-forced')
# #s.autoscale(False)
#     ax[(i-1)/32][(i-1)%32].imshow(layer0[i-1])#,cmap = cm.BLUE,interpolation='bilinear')
#     ax[(i-1)/32][(i-1)%32].autoscale(True)
#     ax[(i-1)/32][(i-1)%32].set_ylim([0,5])
#
# plt.show()


#X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

#layers = lasagne.layers.get_all_layers(network)
#layercounter = 0
#for l in layers:
#    if 'Conv2DLayer' in str(type(l)):
#        plot_conv_activity(l, X_train[:1], figsize=(5,5))
#        plt.show()