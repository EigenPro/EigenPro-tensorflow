from keras import backend as K

def rbf_with_norm(X, Y, XX, YY, gamma):
	XY = K.dot(X, K.transpose(Y))
	d2 = K.reshape(XX, (K.shape(X)[0], 1)) + K.reshape(YY, (1, K.shape(Y)[0])) - 2 * XY
	G = K.exp(-gamma * K.clip(d2, 0, None))
	return G

def rbf(X, Y, gamma):
	""" Radial basis kernel.
	
	Arguments:
		X: of shape (n_sample, n_feature).
		Y: of shape (n_center, n_feature).
		gamma: shape parameter.
	
	Returns:
		kernel matrix of shape (n_sample, n_center).
	"""
	XX = K.sum(K.square(X), axis = 1, keepdims=True)
	if X is Y:
		YY = XX
	else:
		YY = K.sum(K.square(Y), axis = 1, keepdims=True)
	return rbf_with_norm(X, Y, XX, YY, gamma)
