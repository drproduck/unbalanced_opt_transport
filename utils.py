import numpy as np

def get_entropy(P):
    logP = np.log(P + 1e-20)
    return -1 * np.sum(logP * P - P)


def get_KL(P, Q):
    log_ratio = np.log(P + 1e-20) - np.log(Q + 1e-20)
    return np.sum(P * log_ratio - P + Q)

# def get_kl(B, a, axis):
#     xB = B.sum(axis=axis).reshape(-1, 1)
#     log_ratio = np.log(xB / a + 1e-20)
#     ret = xB * log_ratio - xB + a
#     return ret.sum()


def dotp(x, y):
    return np.sum(x * y)

def norm1(X):
    return np.sum(np.abs(X))

def supnorm(X):
    return np.max(np.abs(X))

def round_rc(F, r, c):
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
