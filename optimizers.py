from keras import backend as K
from keras.optimizers import Optimizer

from backend_extra import scatter_update


class PSGD(Optimizer):
    """Primal Stochastic gradient descent optimizer.

    Arguments:
        pred_t: tensor. Prediction result.
        index_t: tensor. Mini-batch indices for primal updates.
        eta: float >= 0. Step size.
        eigenpro_f: Map grad tensor to EigenPro component.
    """

    def __init__(self, pred_t, index_t, eta=0.01, eigenpro_f=None, **kwargs):
        super(PSGD, self).__init__(**kwargs)
        self.eta = K.variable(eta, name='eta')
        self.pred_t = pred_t
        self.index_t = index_t
        self.eigenpro_f = eigenpro_f

    def get_updates(self, params, constraints, loss):
        self.updates = []
        grads = self.get_gradients(loss, [self.pred_t])

        eta = self.eta
        index = self.index_t
        eigenpro_f = self.eigenpro_f

        shapes = [K.get_variable_shape(p) for p in params]
        for p, g in zip(params, grads):
            update_p = K.gather(p, index) - eta * g
            new_p = scatter_update(p, index, update_p)
            
            if eigenpro_f:
                new_p = new_p + eta * eigenpro_f(g)

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'eta': float(K.get_value(self.eta))}
        base_config = super(PSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Arguments:
        eta: float >= 0. Step size.
        eigenpro_f: Map grad tensor to EigenPro component.
    """

    def __init__(self, eta=0.01, eigenpro_f=None, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.eta = K.variable(eta, name='eta')
        self.eigenpro_f = eigenpro_f

    def get_updates(self, params, constraints, loss):
        self.updates = []
        grads = self.get_gradients(loss, params)

        eta = self.eta
        eigenpro_f = self.eigenpro_f

        shapes = [K.get_variable_shape(p) for p in params]
        for p, g in zip(params, grads):
            new_p = p - eta * g
            if eigenpro_f:
                new_p = new_p + eta * eigenpro_f(g)
# apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'eta': float(K.get_value(self.eta))}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
