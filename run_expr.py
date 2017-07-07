'''Train kernel methods on the MNIST dataset.
Should have tensorflow (>=1.2.1) and GPU device.
Run command:
	python run_expr.py
'''

from __future__ import print_function
import collections
import keras
import numpy as np
import time

from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K

import utils
import mnist
from backend_extra import hasGPU
from kernels import rbf
from layers import KernelEmbedding, RFF
from optimizers import PSGD, SGD

assert keras.backend.backend() == u'tensorflow', \
       "Requires Tensorflow (>=1.2.1)."
assert hasGPU(), "Requires GPU."


s2 = 25				# variance of Gaussian kernel
bs = 256 			# size of the mini-batch
n_epoch = 10		# number of epochs for training
eta = np.float32(5)	# step size
M = 4800			# (EigenPro) subsample size.
k = 160				# (EigenPro) top-k eigensystem.

gamma = np.float32(1. / (2 * s2)) # shape parameter of Gaussian kernel
num_classes = 10	# number of classes
eta = eta * num_classes # correction due to mse loss

(x_train, y_train), (x_test, y_test) = mnist.load()
n, D = x_train.shape # (n_sample, n_feature)
d = np.int32(n / 2) * 2 # number of random features

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


Trainer = collections.namedtuple('Trainer', ['model', 'x_train', 'x_test'])
trainers = collections.OrderedDict()

# Assemble Pegasos trainer.
input_shape = (D+1,) # n_feature, (sample) index
kernel = lambda x,y: rbf(x, y, gamma)

ix = Input(shape=input_shape, dtype='float32', name='indexed-feat')
x, index = utils.separate_index(ix)	# mini-batch, sample_ids
kfeat = KernelEmbedding(kernel, x_train,
                        input_shape=input_shape)(x)
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
kf, scale = utils.asm_eigenpro_f(
	utils.add_index(x_train),
	lambda x: embed.predict(x, batch_size=1024),
	M, k, 1, in_rkhs=True)
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
input_shape = (D,)
rff_weights = np.float32(np.sqrt(2 * gamma) * np.random.randn(D, d/2))

x = Input(shape=input_shape, dtype='float32', name='feat')
rf = RFF(rff_weights, input_shape=input_shape)(x)
y = Dense(num_classes, input_shape=(d,),
          activation='linear',
          kernel_initializer='zeros',
          use_bias=False)(rf)
model = Model(x, y)

model.compile(loss='mse',
              optimizer=SGD(eta=eta),
              metrics=['accuracy'])
trainers['SGD with random Fourier feature'] = Trainer(
	model=model, x_train = x_train,	x_test=x_test)

# Assemble EigenPro trainer.
embed = Model(x, rf)
f, scale = utils.asm_eigenpro_f(
	x_train, lambda x: embed.predict(x, batch_size=1024),
	M, k, .25)
model = Model(x, y)
model.compile(loss='mse',
              optimizer=SGD(eta=scale*eta, eigenpro_f=f),
              metrics=['accuracy'])
trainers['EigenPro with random Fourier feature'] = Trainer(
	model=model, x_train = x_train,	x_test=x_test)

# Start training.
for name, trainer in trainers.iteritems():
    print("")
    initial_epoch=0
    np.random.seed(1) # Keras uses numpy random number generator
    train_ts = 0 # training time in seconds
    for epoch in [1, 5, 10, 20, 40]:
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
