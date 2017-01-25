import numpy as np
import theano
import lasagne
from cnn import build_cnn
import theano.tensor as T
import time

import scipy

def get_heatmapp(image):
    print "vafan?????"
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

    # get predictions
    get_preds = theano.function([input_var], test_prediction, allow_input_downcast=True)

    with np.load('../cnn/model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    print "vafan??", image.shape, np.max(image)

    width = image.shape[1]
    height = image.shape[2]

    new_im = np.zeros((1, 3, width, height))

    new_im[0,:,:,:] = image[0:3,:,:]

    res = get_preds(new_im)

    print res

    return res

def get_heatmap(image, stride=8, func=None):
    print "vafan?????"
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

    # get predictions
    get_preds = theano.function([input_var], test_prediction, allow_input_downcast=True)

    with np.load('../cnn/model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    print "vafan??", image.shape, np.max(image)


    width = image.shape[1]
    height = image.shape[2]
    part_width = part_height = 64
    #heat_map = np.zeros((3, width, height))

    jumpi = (width-part_width)/stride
    jumpj = (height-part_height)/stride

    heat_map = np.zeros((3, jumpi, jumpj))

    #heat_map = np.zeros((3, jumpi, jumpj))

    print width, height, heat_map.shape, jumpi, jumpj

    #image_parts = np.zeros((  jumpi*jumpj, 3, part_width, part_height))
    new_im = np.zeros((1, 3, part_width, part_height))

    ii = 0
    for i in range(jumpi):
        for j in range(jumpj):
            #image_parts[ii, :, :, :] = image[0:3, stride*i:stride*i + part_width, stride*j:stride*j + part_height]

            #testt = time.time()
            new_im[0, :, :, :] = image[0:3, stride*i:stride*i + part_width, stride*j:stride*j + part_height]

            res = get_preds(new_im)[0]
            #heat_map[0, stride * (ii % jumpi):stride * (ii % jumpi) + stride, stride * int(ii / jumpj):stride * int(ii / jumpj) + stride] = res[0]
            #heat_map[1, stride * (ii % jumpi):stride * (ii % jumpi) + stride, stride * int(ii / jumpj):stride * int(ii / jumpj) + stride] = res[1]
            #heat_map[2, stride * (ii % jumpi):stride * (ii % jumpi) + stride, stride * int(ii / jumpj):stride * int(ii / jumpj) + stride] = res[2]

            heat_map[0, i, j] = res[0]
            heat_map[1, i, j] = res[1]
            heat_map[2, i, j] = res[2]

            #print ("one loop took: ", time.time()-testt)

            ii += 1

    #res = func(image_parts[:, 0:3, :, :])
    #ii = 0
    #for r in res:
        #print r
    #    heat_map[0, stride * (ii % jumpi):stride * (ii % jumpi) + stride, stride * int(ii / jumpj):stride * int(ii / jumpj) + stride] = r[0]
    #    heat_map[1, stride * (ii % jumpi):stride * (ii % jumpi) + stride, stride * int(ii / jumpj):stride * int(ii / jumpj) + stride] = r[1]
    #    heat_map[2, stride * (ii % jumpi):stride * (ii % jumpi) + stride, stride * int(ii / jumpj):stride * int(ii / jumpj) + stride] = r[2]
        # print(ii)
    #    ii += 1

    return scipy.misc.imresize(np.transpose(heat_map, axes=(2, 1, 0)), float(stride), interp='nearest')
    #return heat_map