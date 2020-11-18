import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils import *
from scipy.special import logsumexp

import pdb

def round_rc(F, r, c):
    """
    "Round" matrix F such that its row sums equal r and its column sums equal c
    "Near-linear time approximation algorithms for OT via Sinkhorn iterations"
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


def get_S(C, u, v, eta):
    K = u + v.T - C
    return np.exp(K / eta)


def f_reg_dual(C, u, v, r, c, eta):
    """
    the dual of the entropic-regularized OT
    """
    S = get_S(C, u, v, eta)
    f = dotp(u, r) + dotp(v, c) - eta * np.sum(S)
    return f


def f_reg_primal(C, u, v, r, c, eta):
    """
    the primal of the entropi-regularized OT
    """
    S = get_S(C, u, v, eta)

    return dotp(C, S) - eta * H(S)


def f_unreg(C, u, v, r, c, eta):
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


def sinkhorn(C, r, c, eta=1.0, n_iter=1000, debug=False, early_stop=None):
    """
    log stable OT
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg debug: if True, return a dict of values (primal, dual objectives etc. )
    :arg early_stop: if None, run all n_iter
                    else, must be a float. stop early if constraint < early_stop
    """

    # collect some info
    if debug:
        info = dict()
        info['u'] = []
        info['v'] = []
        info['f_reg_dual'] = []
        info['f_reg_primal'] = []
        info['f_unreg'] = []
        info['constraint_norm'] = []
    else:
        info = None

    def collect_info(C, u, v, r, c, eta):
        info['u'].append(u)
        info['v'].append(v)
        info['f_reg_dual'].append(f_reg_dual(C, u, v, r, c, eta))
        info['f_reg_primal'].append(f_reg_primal(C, u, v, r, c, eta))
        info['f_unreg'].append(f_unreg(C, u, v, r, c, eta))
        info['constraint_norm'].append(norm1_constraint(C, u, v, r, c, eta))

    log_r = np.log(r)
    log_c = np.log(c)

    # initial solutions
    u = np.zeros(r.shape)
    v = np.zeros(c.shape)


    # compute before any updates
    if debug:
        collect_info(C, u, v, r, c, eta)

    for i in range(n_iter):
        K = u + v.T - C
        log_a = logsumexp(K / eta, axis=1, keepdims=True)
        u = eta * (u / eta + log_r - log_a)

        K = u + v.T - C
        log_b = logsumexp(K.T / eta, axis=1, keepdims=True)
        v = eta * (v / eta + log_c - log_b)

        if debug: 
            collect_info(C, u, v, r, c, eta)

        if early_stop is not None:
            rc_diff = norm1_constraint(C, u, v, r, c, eta)
            if rc_diff < 1e-10:
                break

    return u, v, round_rc(get_S(C, u, v, eta), r, c), info
