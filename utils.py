import numpy as np
import pdb

def get_entropy(P):
    logP = np.log(P + 1e-20)
    return -1 * np.sum(logP * P - P)

def get_KL(P, Q):
    log_ratio = np.log(P + 1e-20) - np.log(Q + 1e-20)
    return np.sum(P * log_ratio - P + Q)

def dotp(x, y):
    return np.sum(x * y)

def norm1(X):
    return np.sum(np.abs(X))

def supnorm(X):
    return np.max(np.abs(X))

def normfro(X):
    return np.sqrt(np.sum(X**2))

def round_rc(F, r, c):
    """
    matrix rounding. Taken from "Near-linear time approximation algorithms for OT via Sinkhorn iterations"
    """
    rF = F.sum(axis=1).reshape(-1, 1)
    x = np.minimum(r / rF, 1)
    FF = F * x.reshape(-1, 1)
    cF = FF.sum(axis=0).reshape(-1, 1)
    y = np.minimum(c / cF, 1)
    FFF = FF * y.reshape(1, -1)

    err_r = r - FFF.sum(axis=1).reshape(-1, 1)
    err_c = c - FFF.sum(axis=0).reshape(-1, 1)
    G = FFF + err_r @ err_c.T / norm1(err_r)

    return G

class BaseGradDescent():
    def __init__(self, lr):
        self.lr = lr
        

    def update(self):
        raise NotImplementedError


class VanillaGradDescent(BaseGradDescent):
    def __init__(self, lr):
        super().__init__(lr)


    def update(self, x, delta):
        return x - self.lr * delta


class NesterovGradDescent(BaseGradDescent):
    def __init__(self, lr):
        super().__init__(lr)
        self.lmd = 0
        self.y_old = None

    def update(self, x, delta):
        lmd_1 = (1 + np.sqrt(1 + 4 * self.lmd ** 2)) / 2
        lmd_2 = (1 + np.sqrt(1 + 4 * lmd_1 ** 2)) / 2
        gamma = (1 - lmd_1) / lmd_2
        print(gamma)
        self.lmd = lmd_1

        y = x - self.lr * delta
        if self.y_old is None:
            x = (1 - gamma) * y + gamma * x
        else:
            x = (1 - gamma) * y + gamma * self.y_old
        self.y_old = y

        return x


class BacktrackGradDescent(BaseGradDescent):
    def __init__(self, lr, beta, eval_func, update_func):
        super().__init__(lr)
        self.beta = beta
        self.eval_func = eval_func
        self.update_func = update_func


    def update(self, x_old, delta, f_old):
        alpha = self.lr
        x_new = self.update_func(x_old, delta)
        f_new = self.eval_func(x_new)
        while (not np.isfinite(f_new)) or (not f_new <= f_old - alpha * self.beta * np.sum(delta**2)):
            alpha /= 2
            x_new = self.update_func(x_old, delta)
            f_new = self.eval_func(x_new)

        return x_new, f_new

