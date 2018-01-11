'''Train kernel methods on the MNIST dataset.
Should have tensorflow (>=1.2.1) and GPU device.
Run command:
	python run_expr.py
'''

from __future__ import print_function

import argparse
import collections
import keras
import numpy as np
import time
import warnings

from distutils.version import StrictVersion
from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K

import kernels
import mnist
import utils

from backend_extra import hasGPU
from layers import KernelEmbedding, RFF
from optimizers import PSGD, SGD

assert StrictVersion(keras.__version__) >= StrictVersion('2.0.8'), \
       "Requires Keras (>=2.0.8)."

if StrictVersion(keras.__version__) > StrictVersion('2.0.8'):
    warnings.warn('\n\nEigenPro-tensorflow has been tested with Keras 2.0.8. '
                   'If the\ncurrent version (%s) fails, ' 
                   'switch to 2.0.8 by command,\n\n'
                   '\tpip install Keras==2.0.8\n\n' %(keras.__version__), Warning)

assert keras.backend.backend() == u'tensorflow', \
       "Requires Tensorflow (>=1.2.1)."
assert hasGPU(), "Requires GPU."

parser = argparse.ArgumentParser(description='Run EigenPro tests.')
parser.add_argument('--kernel', type=str, default='Gaussian',
                    help='kernel function (e.g. Gaussian, Laplace, and Cauchy)')
args = parser.parse_args()
args_dict = vars(args)


# Set the hyper-parameters.
bs = 256            # size of the mini-batch
M = 4800            # (EigenPro) subsample size
k = 160             # (EigenPro) top-k eigensystem

num_classes = 10	    # number of classes

(x_train, y_train), (x_test, y_test) = mnist.load()
n, D = x_train.shape    # (n_sample, n_feature)
d = np.int32(n>>1) * 2 # number of random features

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if args_dict['kernel'] == 'Gaussian':
    s = 5   # kernel bandwidth
    kernel = lambda x,y: kernels.Gaussian(x, y, s)

elif args_dict['kernel'] == 'Laplace':
    s = np.float32(10)
    kernel = lambda x,y: kernels.Laplace(x, y, s)

elif args_dict['kernel'] == 'Cauchy':
    s = np.sqrt(40, dtype=np.float32)
    kernel = lambda x,y: kernels.Cauchy(x, y, s)

else:
    raise Exception("Unknown kernel function - %s. \
                     Try Gaussian, Laplace, or Cauchy"
                    % args_dict['kernel'])

trainers = collections.OrderedDict()
Trainer = collections.namedtuple('Trainer', ['model', 'x_train', 'x_test'])


# Calculate step size and (Primal) EigenPro preconditioner.
kf, scale, s0 = utils.asm_eigenpro_f(
    x_train, kernel, M, k, 1, in_rkhs=True)
eta = np.float32(1.5 / s0) # 1.5 / s0
eta = eta * num_classes # correction due to mse loss

# Assemble Pegasos trainer.
input_shape = (D+1,) # n_feature, (sample) index
ix = Input(shape=input_shape, dtype='float32', name='indexed-feat')
x, index = utils.separate_index(ix)	# features, sample_id
kfeat = KernelEmbedding(kernel, x_train,
                        input_shape=(D,))(x)
y = Dense(num_classes, input_shape=(n,),
          activation='linear',
          kernel_initializer='zeros',
          use_bias=False)(kfeat)

model = Model(ix, y)
model.compile(loss='mse',
              optimizer=PSGD(pred_t=y, index_t=index, eta=eta),
              metrics=['accuracy'])
trainers['Pegasos'] = Trainer(model=model,
                              x_train = utils.add_index(x_train),
                              x_test=utils.add_index(x_test))

# Assemble kernel EigenPro trainer.
embed = Model(ix, kfeat)
y = Dense(num_classes, input_shape=(n,),
          activation='linear',
          kernel_initializer='zeros',
          use_bias=False)(kfeat)
model = Model(ix, y)
model.compile(loss='mse',
              optimizer=PSGD(pred_t=y,
                             index_t=index,
                             eta=scale*eta,
                             eigenpro_f=lambda g: kf(g, kfeat)),
              metrics=['accuracy'])
trainers['Kernel EigenPro'] = Trainer(model=model,
                              x_train = utils.add_index(x_train),
                              x_test=utils.add_index(x_test))


# Assemble SGD trainer.
rff_weights = np.float32(       # for Gaussian kernel
    np.sqrt(2. / (2 * 5 ** 2))  # s = 5
    * np.random.randn(D, d>>1))
input_shape = (D,)
x = Input(shape=input_shape, dtype='float32', name='feat')
rf_f = RFF(rff_weights, input_shape=input_shape)
y = Dense(num_classes, input_shape=(d,),
          activation='linear',
          kernel_initializer='zeros',
          use_bias=False)(rf_f(x))
model = Model(x, y)

model.compile(loss='mse',
              optimizer=SGD(eta=eta),
              metrics=['accuracy'])
trainers['SGD with random Fourier feature'] = Trainer(
	model=model, x_train = x_train,	x_test=x_test)

# Assemble EigenPro trainer.
f, scale, _ = utils.asm_eigenpro_f(
	x_train, rf_f, M, k, .25)
y = Dense(num_classes, input_shape=(d,),
          activation='linear',
          kernel_initializer='zeros',
          use_bias=False)(rf_f(x))
model = Model(x, y)
model.compile(loss='mse',
              optimizer=SGD(eta=scale*eta, eigenpro_f=f),
              metrics=['accuracy'])
trainers['EigenPro with random Fourier feature'] = Trainer(
	model=model, x_train = x_train,	x_test=x_test)

# Start training.
for name, trainer in trainers.items():
    print("")
    initial_epoch=0
    np.random.seed(1) # Keras uses numpy random number generator
    train_ts = 0 # training time in seconds
    for epoch in [1, 2, 5, 10, 20, 40]:
        start = time.time()
        trainer.model.fit(
			trainer.x_train, y_train,
			batch_size=bs, epochs=epoch, verbose=0,
			validation_data=(trainer.x_test, y_test),
			initial_epoch=initial_epoch)
        train_ts += time.time() - start
        tr_score = trainer.model.evaluate(trainer.x_train, y_train, verbose=0)
        te_score = trainer.model.evaluate(trainer.x_test, y_test, verbose=0)
        print("%s\t\ttrain error: %.2f%%\ttest error: %.2f%% (%d epochs, %.2f seconds)" %
              (name, (1 - tr_score[1]) * 100, (1 - te_score[1]) * 100, epoch, train_ts))
        initial_epoch = epoch
