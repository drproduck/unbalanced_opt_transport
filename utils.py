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
