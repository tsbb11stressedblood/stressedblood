import numpy as np
import lasagne
from cnn import build_cnn
from cnn import iterate_minibatches
from cnn import load_dataset
import theano
import theano.tensor as T


input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

print("Building model and compiling functions...")
network = build_cnn(input_var)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)



val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

# Create neural network model (depending on first command line parameter)


with np.load('model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)




test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 50, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))