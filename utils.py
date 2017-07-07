import numpy as np

from keras import backend as K
from keras.layers import Lambda

def add_index(X):
    """Append sample index as the last feature to data matrix.

    Arguments:
        X: matrix of shape (n_sample, n_feat).

    Returns:
        matrix of shape (n_sample, n_feat+1).
    """
    inx = np.reshape(np.arange(X.shape[0]), (-1, 1))
    return np.hstack([X, inx])

def separate_index(IX):
    """Separate the index feature from the indexed tensor matrix.

    Arguments:
        IX: matrix of shape (n_sample, n_feat+1).

    Returns:
        X: matrix of shape (n_sample, n_feat).
        index: vector of shape (n_sample,).
    """
    X = Lambda(lambda x: x[:, :-1])(IX)
    index = Lambda(lambda x: x[:, -1])(IX)
    return X, K.cast(index, dtype='int32')

def rsvd(X, phi, M, k):
    """Subsample randomized SVD based on
    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
    "Finding structure with randomness: Probabilistic algorithms
        for constructing approximate matrix decompositions."
    SIAM review 53.2 (2011): 217-288.

    Arguments:
        X:      feature matrix of shape (n, D).
        phi:    feature map: R^D -> R^d.
        M:      subsample size.
        k:      top eigensystem.

    Returns:
        s:  (k,)    top-k eigenvalues of phi(X).
        V:  (d, k)  top-k eigenvectors of phi(X).
        sk: (k+1)-th largest eigenvalue of phi(X).
    """
    n, _ = X.shape
    index = np.random.choice(n, M, replace=False)
    A = phi(X[index])

    d = A.shape[1]
    p = min(2 * (k+1), d)
    R = np.random.randn(d, p)
    Y = np.dot(A, R)
    W, _ = np.linalg.qr(Y)
    B = np.dot(W.T, A)
    _, S1, VT1 = np.linalg.svd(B, full_matrices=False)

    s = np.sqrt(n / M) * S1[:k]
    sk = np.sqrt(n / M) * S1[k]
    V = VT1[:k].T
    return s, V, sk

def asm_eigenpro_f(feat, phi, M, k, tau, in_rkhs=False, svd_f=rsvd):
    """Assemble eigenpro map and calculate step size scale factor
    such that the update rule,
        p <- p - eta * g
    becomes,
        p <- p - scale * eta * (g - f(g))

    Arguments:
        feat:   feature matrix.
        phi:    feature map.
        M:      subsample size.
        k:      top-k eigensystem for eigenpro.
        tau:    damping factor.

    Returns:
        f:      tensor function.
        scale:  factor that rescales step size.
    """

    _s, _V, _sk = svd_f(feat, phi, M, k)
    s = K.constant(_s)
    V = K.constant(_V)
    sk = K.constant(_sk)

    if in_rkhs:
        scale = np.sqrt(_s[0] / _sk, dtype='float32')
        D = (1 - K.sqrt(tau * sk / s)) / s
        f = lambda g, kfeat: K.dot(
            V * D, K.dot(K.transpose(K.dot(kfeat, V)), g))
    else:
        scale = _s[0] / _sk
        D = 1 - tau * sk / s
        f = lambda g: K.dot(V * D, K.dot(K.transpose(V), g))

    return f, scale
