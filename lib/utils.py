import numpy as np
import pdb

def H(P):
    """
    the entropy
    """
    logP = np.log(P + 1e-20)
    return -1 * np.sum(logP * P - P)


def KL(P, Q):
    """
    The KL divergence
    """

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


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
        shape = [-1, 1]
    axis: int
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        cssv = np.cumsum(U, axis=1) - z
        ind = np.arange(n_features) + 1.
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / (1. * rho)
        res = np.maximum(V - theta.reshape(-1, 1), 0)
        res[~np.isfinite(res)] = 0
        return res

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

