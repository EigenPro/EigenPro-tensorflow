import numpy as np

from keras import backend as K

def D2(X, Y):
    """ Calculate the pointwise (squared) distance.
    
    Arguments:
    	X: of shape (n_sample, n_feature).
    	Y: of shape (n_center, n_feature).
    
    Returns:
    	pointwise distances (n_sample, n_center).
    """
    XX = K.sum(K.square(X), axis = 1, keepdims=True)
    if X is Y:
        YY = XX
    else:
        YY = K.sum(K.square(Y), axis = 1, keepdims=True)
    XY = K.dot(X, K.transpose(Y))
    d2 = K.reshape(XX, (K.shape(X)[0], 1)) \
       + K.reshape(YY, (1, K.shape(Y)[0])) \
       - 2 * XY
    return d2

def Gaussian(X, Y, s):
    """ Gaussian kernel.
    
    Arguments:
    	X: of shape (n_sample, n_feature).
    	Y: of shape (n_center, n_feature).
    	s: kernel bandwidth.
    
    Returns:
    	kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0
    
    d2 = D2(X, Y)
    gamma = np.float32(1. / (2 * s ** 2))
    G = K.exp(-gamma * K.clip(d2, 0, None))
    return G

def Laplace(X, Y, s):
    """ Laplace kernel.
    
    Arguments:
    	X: of shape (n_sample, n_feature).
    	Y: of shape (n_center, n_feature).
    	s: kernel bandwidth.
    
    Returns:
    	kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0
    
    d2 = K.clip(D2(X, Y), 0, None)
    d = K.sqrt(d2)
    G = K.exp(- d / s)
    return G

def Cauchy(X, Y, s):
    """ Cauchy kernel.
    
    Arguments:
    	X: of shape (n_sample, n_feature).
    	Y: of shape (n_center, n_feature).
    	s: kernel bandwidth.
    
    Returns:
    	kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0
    
    d2 = D2(X, Y)
    s2 = np.float32(s**2)
    G = 1 / K.exp( 1 + K.clip(d2, 0, None) / s2)
    return G
