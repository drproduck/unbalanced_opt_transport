import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.special import logsumexp

def get_S(C, u, v, eta):
    K = u + v.T - C
    return np.exp(K / eta)


def f(C, u, v, r, c, eta):
    """
    the dual of the entropic-regularized balanced OT
    """
    S = get_S(C, u, v, eta)
    f = dotp(u, r) + dotp(v, c) - eta * np.sum(S)
    return f


def f_primal(C, u, v, eta):
    S = get_S(C, u, v, eta)
    unreg_val = dotp(C, S)

    entropy = H(S)

    return unreg_val - eta * entropy


def unreg_f(C, u, v, eta):
    """
    the unregularized objective with solutions u, v
    """
    S = get_S(C, u, v, eta)

    return dotp(C, S)


def norm1_constraint(C, u, v, r, c, eta):
    S = get_S(C, u, v, eta)
    a = S.sum(axis=1).reshape(-1, 1)
    b = S.sum(axis=0).reshape(-1, 1)
    return norm1(a - r) + norm1(b - c)


def sinkhorn_ot(C, r, c, eta=1.0, n_iter=100, duals=False):
    """
    log stable balanced OT
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    """

    log_r = np.log(r)
    log_c = np.log(c)


    # initial solutions
    u = np.zeros(r.shape)
    v = np.zeros(c.shape)


    for i in range(n_iter):
        K = u + v.T - C
        log_a = logsumexp(K / eta, axis=-1, keepdims=True)
        u = eta * (u / eta + log_r - log_a)

        K = u + v.T - C
        log_b = logsumexp(K.T / eta, axis=-1, keepdims=True)
        v = eta * (v / eta + log_c - log_b)


        rc_diff = norm1_constraint(C, u, v, r, c, eta)

        if rc_diff < 1e-10:
            break

    S = get_S(C, u, v, eta)

    if duals:
        return S, u, v
    else:
        return S

