from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class KernelEmbedding(Layer):
    """ Generate kernel features.

    Arguments:
        kernel_f:   kernel function k(x, y).
        centers:    matrix of shape (n_center, n_feature).
    """

    def __init__(self, kernel_f, centers, **kwargs):
        self.kernel_f = kernel_f
        self._centers = centers
        self.n_center = centers.shape[0]
        super(KernelEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers', 
                                       shape=self._centers.shape,
                                       initializer=(lambda shape: self._centers),
                                       trainable=False)
        super(KernelEmbedding, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        embed = self.kernel_f(x, self.centers)
        return embed

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_center)


def rff(X, W):
    """Calculate random Fourier features according to paper,
      'Random Features for Large-Scale Kernel Machines'.

    Arguments:
        X: data matrix of shape (n, D).
        W: weight matrix of shape (D, d).

    Returns:
        feature matrix of shape (n, d).
    """

    d = K.get_variable_shape(W)[1]
    dot = K.dot(X, W) # of shape (n, d)
    RF = K.concatenate([K.cos(dot), K.sin(dot)], axis=1) / np.sqrt(d, dtype='float32')
    return RF


class RFF(Layer):
    """ Generate random Fourier features.

    Arguments:
        weights: of shape (D, d).
    """

    def __init__(self, weights, **kwargs):
        self._weights = weights
        self.d = weights.shape[1]
        super(RFF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='rff-weight', 
                                 shape=self._weights.shape,
                                 initializer=(lambda shape: self._weights),
                                 trainable=False)
        super(RFF, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        embed = rff(x, self.W)
        return embed

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.d)
