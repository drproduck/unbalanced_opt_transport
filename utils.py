import numpy as np

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

    def update(x, delta):
        return x - self.lr * delta


class NesterovGradDescent(BaseGradDescent):
    def __init__(self, lr):
        super().__init__(lr)
        self.lmd = 0
        self.old_y = None

    def update(x, delta, subset=None):
        new_lmd = (1 + np.sqrt(1 + 4 * self.lmd ** 2)) / 2
        gamma = (1 - self.lmd) / new_lmd
        self.lmd = new_lmd

        y = x - self.lr * delta
        if self.old_y is None:
            x = (1 - gamma) * y + gamma * x
        else:
            x = (1 - gamma) * y + gamma * self.old_y
        self.old_y = y

        return x


# class BacktrackGradDescent(BaseGradDescent):
#     def __init__(self, lr):
