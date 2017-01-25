#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time
import cv2
import numpy as np

try:
    import theano
except:
    pass

import theano.tensor as T
import matplotlib.image as mpimg
import fnmatch
import scipy

try:
    import lasagne
except:
    pass
from PIL import Image
import math
#import lasagne.layers.dnn
from scipy.misc import imresize, imrotate

use_64 = True #else 32


import matplotlib.pyplot as plt
# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def resize_image(img):
    if img.shape[0] > 64:
        if img.shape[1] > 64:
            pass
        else: #0 > 64, 1 < 64
            pass

    elif img.shape[1] > 64:
        pass

    else:
        imgaa = np.pad(img, ((0, 0), (0, 64 - img.shape[0]), (0, 64 - img.shape[1])), 'constant', constant_values=(0, 0))

    return imgaa


def load_dataset():

    def load_images_and_labels(folder):
        from os import listdir
        from os.path import isfile, join
        #files = [f for r,d,f in os.walk(folder) if isfile(join(folder, f))]

        labels = []
        i = 0
        result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames if
                  os.path.splitext(f)[1] == '.png']

        num = len(result)

        rotations = 10


        if use_64:
            images = np.zeros((num * (rotations+1), 3, 64, 64))
        else:
            images = np.zeros((num * 2, 3, 32, 32))

        for f in result:
                #img = Image.open(folder + filename, 'r').convert('LA')
            #img = mpimg.imread( f, 'r')[:,:,2]*.299 + mpimg.imread( f, 'r')[:,:,1]*.587 + mpimg.imread( f, 'r')[:,:,0]*.114
            #img = mpimg.imread(f, 'r')[:, :, 0:1]
            img = mpimg.imread(f, 'r') / np.float32(256.0)



            #subtract mean
            #img[:,:,0] = img[:,:,0] - img[:,:,0].mean()
            #img[:, :, 1] = img[:, :, 1] - img[:, :, 1].mean()
            #img[:, :, 2] = img[:, :, 2] - img[:, :, 2].mean()

            img = img - img.mean()

            #img = img[:, :, 0:1]*.05 + img[:, :, 1:2]*.05 + img[:, :, 2:3]*.9

            img = np.transpose(img, axes=(2,1,0))
            #print ("imgshape:", img.shape)
            imga = np.zeros((3, img.shape[1], img.shape[2]), dtype='float32')
            imga[:,:,:] = img
                #imga = np.transpose(img)

                #imga = np.array( img.getdata(),
                #        np.uint8)
                #plt.imshow(imga)
                #plt.interactive(False)
                #print("YEEEAAHHH")
            #print (imga.shape)

                #imga.resize([1, 64, 64])

            #imgaa = resize_image(imga)

            if use_64:
                if img.shape[1] < 64 or img.shape[2] < 64:
                    if img.shape[1] > img.shape[2]:
                        factor = 64.0/img.shape[2]+.1
                    else:
                        factor = 64.0/img.shape[1]+.1

                    imgb = scipy.misc.imresize(imga, factor)

                    imga = np.transpose(imgb, axes=(2, 1, 0))

                    #print(imgb.shape)

                if imga.shape[1] > 64:
                    imga = imga[:,0:64,:]
                if imga.shape[2] > 64:
                    imga = imga[:,:,0:64]
                print(imga.shape)

            else: #use 32
                if imga.shape[1] < 32:
                    imga = np.pad(imga, ((0, 0), (0, 32 - img.shape[1]), (0, 0)), 'constant',
                                  constant_values=(0, 0))

                if imga.shape[2] < 32:
                    imga = np.pad(imga, ((0, 0), (0, 0), (0, 32 - img.shape[2])), 'constant',
                                  constant_values=(0, 0))

                if imga.shape[1] < 32 and imga.shape[2] < 32:
                    imga = np.pad(imga, ((0, 0), (0, 32 - img.shape[1]), (0, 32 - img.shape[2])),
                                  'constant', constant_values=(0, 0))

                if imga.shape[1] > 32:
                    imga = imga[:, 0:32, :]
                if imga.shape[2] > 32:
                    imga = imga[:, :, 0:32]


                    #else:
            #    print(f)

                #imgaa = imga[:,:,:]
                #imgaa = np.zeros((1,64,64))
            #print(f)
            #imgaa = imga[:,0:32, 0:32]

            #print ("imgaa: ", imgaa.shape)

            #plt.imshow(imgaa[0,:,:])
            #plt.show()

            images[i,:,:,:] = imga
            i += 1
            if 'blue' in f:
                labels.append(0)
            elif 'green' in f:
                labels.append(1)
            elif 'red' in f:
                labels.append(0)
            elif 'pink' in f: #crap
                labels.append(3)
            else:
                print ("SHOULDNT HAPPEN!!!")

        #images = np.reshape(-1, 1, 128, 128)

        #print("images_shape:", images.shape)

        print("num: ", num)
        #perturb images?
        #perturbed_images = images.copy()
        for i in range(rotations):
            perturbed_images1 = np.zeros((num, 3, 64, 64))
            for ii in range(num):
                perturbed_images1[ii,:,:,:] = np.transpose(perturb(images[ii,:,:,:], 1, (i+1)*180/rotations, 0, 0), axes=(2,1,0))

            images[num*i + num:num*i + num + num, :, :, :] = perturbed_images1

        #perturbed_images2 = np.zeros((num, 3, 64, 64))
        #for im in perturbed_images2:
        #    im = perturb(im, 1, 180, 0, 0)

        #images[num*2:num*2 + num, :, :, :] = perturbed_images2

        return images/np.float32(256), labels*rotations, num*rotations


    def perturb(img, scale, angle, skew, mirror):
        img2 = img.copy()
        if scale is not 1:
            img2 = imresize(img2, scale)
        if angle is not 0:
            img2 = imrotate(img2, angle)
        if skew is not 0:
            pass
        if mirror:
            img2 = np.fliplr(img2)

        #plt.imshow(img2)
        #plt.show()

        return img2

    #X_train, y_train = load_images_and_labels('../learning_images/9W/2015-10-15 18.17-2/')
    X_train, y_train, num = load_images_and_labels('../learning_images/all/')
    #X_train, y_train, num = load_images_and_labels('../learning_images/all_littlecrap/')
    #X_train, y_train, num = load_images_and_labels('../learning_images/small_test/')
    #X_train, y_train, num = load_images_and_labels('../learning_images/9W_littlecrap/')

    print (y_train)
    #X_test, y_test = load_images_and_labels('../learning_images/9W/2015-10-15 19.02_1/')
    #X_test, y_test = load_images_and_labels('../learning_images/12W/')


    # We reserve the last n training examples for validation.


    #for i in range(200):
    #    plt.imshow(X_test[i,0,:,:], cmap=plt.cm.gray)
        # plt.imshow(dataset['train']['X'][0][0], interpolation='nearest', cmap=plt.cm.gray)
    #    plt.show()

    #print (len(y_test))
    #y_test = [0 for i in range(200)]
    #y_test = np.zeros((200,1))

    trainp = 5/8.0
    valp = 1/4.0
    testp = 1/8.0


    print(num)
    X_test = X_train[int(num*(trainp+valp)):num]
    y_test = y_train[int(num*(trainp+valp)):num]

    X_val = X_train[int(num*trainp):int(num*(trainp+valp))]
    y_val = y_train[int(num*trainp):int(num*(trainp+valp))]

    y_train = y_train[0:int(num*trainp)]
    X_train = X_train[0:int(num*trainp)]

    print("0: ", y_train.count(0))
    print("1: ", y_train.count(1))
    print("2: ", y_train.count(2))
    print("3: ", y_train.count(3))

    print("X_train: ", X_train.shape)
    print("y_train: ", len(y_train))

    print("X_test: ", X_test.shape)
    print("y_test: ", len(y_test))

    print("X_val: ", X_val.shape)
    print("y_val: ", len(y_val))

    #X_train = np.array(X_train)#.resize((64, 64))
    #print("X_train:", X_train.shape)
    #X_val = X_val.reshape((-1, 1, 64, 64))
    #X_test = X_test.reshape((-1, 1, 64, 64))

    # We just return all the arrays in order, as expected in main().
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    if use_64:
        network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                        input_var=input_var)
    else:
        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                            input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


###############################


    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())


    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


###############################



    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)


    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=4,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    #print(len(inputs), len(targets))
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        #if shuffle:
        #    excerpt = indices[start_idx:start_idx + batchsize]
        #else:
        excerpt = slice(start_idx, start_idx + batchsize)
        #print(len(inputs), len(targets), len(excerpt))
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='cnn', num_epochs=8000):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    #dataset = {
    #    'train': {'X': X_train, 'y': y_train},
    #    'valid': {'X': X_val, 'y': y_val}}
    # Plot an example digit with its label
    #plt.imshow(dataset['train']['X'][0][0], interpolation='nearest', cmap=plt.cm.gray)
    #plt.title("Label: {}".format(dataset['train']['y'][0]))
    #plt.gca().set_axis_off()
    #plt.show()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    print  ("pred: ", prediction)
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

    # Finally, launch the training loop.
    print("Starting training...")

    accs = []
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, 128, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            #print(train_batches)

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 128, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        acc = val_acc / val_batches * 100
        accs.append(acc)
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0

    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 128, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)

        print ("err, acc: ", err, acc)

        #if err:
        #    print (inputs)
        #    print (inputs.shape)
        #    plt.imshow(inputs[0,0,:,:])
        #    plt.show()
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))



    #test_image = mpimg.imread('../fake_areas/9W/1.png', 'r')/np.float32(256.0)
    #test_image = mpimg.imread('../nice_areas/9W/2015-10-15 18.06_1.png', 'r') / np.float32(256.0)
    test_image = mpimg.imread('../nice_areas/9W/512x512/2015-10-15 18.06_1.png', 'r') / np.float32(256.0)

    heat_map = np.zeros((3,512,512))

    #print ("before:", test_image.shape)
    test_image = np.transpose(test_image, axes=(2, 1, 0))
    #print("after:", test_image.shape)

    if use_64:
        images = np.zeros((4096, 3, 64, 64))
    else:
        images = np.zeros((64, 3, 32, 32))
    ii = 0
    for i in range(56):
        for j in range(56):
            if use_64:
                #images[ii,:,:,:] = test_image[0:3, 64*i:64*i+64, 64*j:64*j+64]
                images[ii, :, :, :] = test_image[0:3, 8 * i:8 * i + 64, 8 * j:8 * j + 64]
            else:
                images[ii, :, :, :] = test_image[0:3, 32 * i:32 * i + 32, 32 * j:32 * j + 32]
            ii += 1

    #images[0, :, :, :] = test_image1[0:3,:,:]
    #images[1, :, :, :] = test_image2[0:3,:,:]

    if use_64:
        testaa = get_preds(images[0:2809,0:3,:,:])
    else:
        testaa = get_preds(images[0:255, 0:3, :, :])

    ii = 0
    for t in testaa:
        heat_map[0, 8 * (ii % 56):8 * (ii % 56) + 8, 8 * int(ii / 56):8 * int(ii / 56) + 8] = t[0]
        heat_map[1, 8 * (ii % 56):8 * (ii % 56) + 8, 8 * int(ii / 56):8 * int(ii / 56) + 8] = t[1]
        heat_map[2, 8 * (ii % 56):8 * (ii % 56) + 8, 8 * int(ii / 56):8 * int(ii / 56) + 8] = t[2]
        #print(ii)
        ii += 1

    plt.imshow(np.transpose(heat_map, axes=(1, 2, 0)))
    #plt.title("Label: {}".format(testaa[i]))
    #plt.show()
    plt.figure()
    plt.imshow(np.transpose(-np.reciprocal(np.log10(heat_map)), axes=(1, 2, 0)))
    # plt.title("Label: {}".format(testaa[i]))
    plt.show()


    # Optionally, you could now dump the network weights to a file like this:
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    print(testaa)
    print(lasagne.__version__)
    #print ("TESTA: ", ['{0:.2}'.format(i) for i in testaa ]  )

    new = np.transpose(images, axes=(0,3,2,1))
    for i in range(255):
        if testaa[i][2] > .4:
            plt.imshow(new[i,:,:,:])
            plt.title("Label: {}".format(testaa[i]))
            plt.show()

    #print (inputs)



    #plot accuracies
    plt.plot(accs)
    plt.show()


    layers = lasagne.layers.get_all_layers(network)



if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)