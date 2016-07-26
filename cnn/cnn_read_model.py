
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
import lasagne.layers.dnn
from scipy.misc import imresize, imrotate
from cnn import build_cnn

import matplotlib.pyplot as plt

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


test_image = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 17.44_1.png', 'r') / np.float32(256.0)

heat_map = np.zeros((3,512,512))

    #print ("before:", test_image.shape)
test_image = np.transpose(test_image, axes=(2, 1, 0))
    #print("after:", test_image.shape)

images = np.zeros((4096, 3, 64, 64))

ii = 0

for i in range(56):
    for j in range(56):
        #images[ii,:,:,:] = test_image[0:3, 64*i:64*i+64, 64*j:64*j+64]
        images[ii, :, :, :] = test_image[0:3, 8 * i:8 * i + 64, 8 * j:8 * j + 64]
        ii += 1

testaa = get_preds(images[0:2809, 0:3, :, :])

ii = 0
for t in testaa:
    heat_map[0, 8 * (ii % 56):8 * (ii % 56) + 8, 8 * int(ii / 56):8 * int(ii / 56) + 8] = t[0]
    heat_map[1, 8 * (ii % 56):8 * (ii % 56) + 8, 8 * int(ii / 56):8 * int(ii / 56) + 8] = t[1]
    heat_map[2, 8 * (ii % 56):8 * (ii % 56) + 8, 8 * int(ii / 56):8 * int(ii / 56) + 8] = t[2]
    #print(ii)
    ii += 1

plt.imshow(np.transpose(heat_map, axes=(1, 2, 0)))
plt.figure()
plt.imshow(np.transpose(-np.reciprocal(np.log10(heat_map)), axes=(1, 2, 0)))
# plt.title("Label: {}".format(testaa[i]))
plt.show()