import numpy as np
import tensorflow as tf
import time

from keras import backend as K
from keras.layers import Lambda, Input
from keras.models import Model
from sklearn.decomposition import TruncatedSVD

from layers import KernelEmbedding

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

def asm_eigenpro_f(feat, phi, M, k, tau, in_rkhs=False):
    """Assemble eigenpro map and calculate step size scale factor
    such that the update rule,
        p <- p - eta * g
    becomes,
        p <- p - scale * eta * (g - f(g))

    Arguments:
        feat:   feature matrix.
        phi:    feature map or kernel function.
        M:      subsample size.
        k:      top-k eigensystem for eigenpro.
        tau:    damping factor.

    Returns:
        f:      tensor function.
        scale:  factor that rescales step size.
        s0:     largest eigenvalue.
    """

    start = time.time()
    n, D = feat.shape
    x = Input(shape=(D,), dtype='float32', name='feat')
    if in_rkhs:
        if n >= 10**5:
            _s, _V = nystrom_kernel_svd(feat, phi, M, k) # phi is k(x, y)
        else:
            kfeat = KernelEmbedding(phi, feat,
                                    input_shape=(D, ))(x)
            model = Model(x, kfeat)
            fmap = lambda _x: model.predict(_x, batch_size=1024)
            _s, _V, _sk = rsvd(feat, fmap, M, k) # phi is a feature map
    else:
        model = Model(x, phi(x))
        fmap = lambda _x: model.predict(_x, batch_size=1024)
        _s, _V, _sk = rsvd(feat, fmap, M, k) # phi is a feature map
    _s, _sk, _V = _s[:k], _s[-1], _V[:, :k]
    print("SVD time: %.2f, Eigenvalue ratio: %.2f" %
          (time.time() - start, _s[0] / _sk))

    s = K.constant(_s)
    V = K.constant(_V)
    sk = K.constant(_sk)

    if in_rkhs:
        scale = np.sqrt(_s[0] / _sk, dtype='float32')
        D = (1 - K.sqrt(tau * sk / s)) / s
        f = lambda g, kfeat: K.dot(
            V * D, K.dot(K.transpose(K.dot(kfeat, V)), g))
        s0 = 2 * _s[0] / n
    else:
        scale = np.float32(_s[0] / _sk)
        D = 1 - tau * sk / s
        f = lambda g: K.dot(V * D, K.dot(K.transpose(V), g))
        s0 = np.float32(_s[0] / np.sqrt(n))

    return f, scale, s0


def GramSchmidtProcess(A):
    """Gram-Schmidt Orthonormalization.
    
    Arguments:
        A: matrix of shape (n_vector, n_feature).

    Returns:
        B: orthonormalized matrix.
    """
    n, d = A.shape
    assert n <= d

    A = tf.Variable(A, name='GS-A')
    cursor = tf.placeholder(tf.int32)
    a = tf.reshape(A[cursor], (1, -1))
    new_a = a - tf.matmul(tf.matmul(a, tf.transpose(A[:cursor])),
                          A[:cursor])
    na = tf.reshape(new_a / tf.norm(new_a), (-1,))
    update = tf.scatter_update(A, cursor, na)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index in np.arange(n):
            sess.run(update, feed_dict={cursor: index})
        A = sess.run(A)
    return A

def nystrom_kernel_svd(X, kernel_f, m, k, bs=512):
    """Compute top eigensystem of kernel matrix using Nystrom method.

    Arguments:
        X: data matrix of shape (n_sample, n_feature).
        kernel_f: kernel tensor function k(X, Y).
        m: subsample factor.
        k: top-k eigensystem.
        bs: batch size.

    Returns:
        s: top eigenvalues of shape (k).
        U: top eigenvectors of shape (n_sample, k).
    """

    n, d = X.shape
    m = min(m, n)
    inx = np.random.permutation(n)[:m]
    Xm = X[inx]

    # Assemble kernel function evaluator.
    input_shape = (d, )
    x = Input(shape=input_shape, dtype='float32',
              name='nystrom-kernel-feat')
    K_t  = KernelEmbedding(kernel_f, Xm)(x)
    kernel_tf = Model(x, K_t)
    
    Kmm = kernel_tf.predict(Xm, batch_size=bs)
    D = np.float32(np.ones((m, 1)) * np.sqrt(n) / np.sqrt(m))
    W = D * Kmm * D.T

    U1r, sr, _ = np.linalg.svd(W)
    s = sr[:k]
    DU1 = K.variable(D * U1r[:, :k])
    
    U2_t = Lambda(lambda _K: K.dot(_K, DU1))(K_t)
    U2_tf = Model(x, U2_t)
    U2 = U2_tf.predict(X, batch_size=bs)
    U = U2 / np.linalg.norm(U2, axis=0, keepdims=True)
    NU = GramSchmidtProcess(U.T).T

    return s, NU
