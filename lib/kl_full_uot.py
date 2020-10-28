import numpy as np
from utils import *
from copy import copy
import numpy 
from scipy.special import logsumexp
from functools import partial
import pdb

np.random.seed(999)


def get_B(C, u, v, eta):
    n, m = C.shape
    K = - C + u + v.T
    return np.exp(K / eta)


def f_dual(B, u, v, r, c, eta, t1, t2):
    """
    the dual of the entropic-regularized unbalanced OT
    """
    f = - eta * np.sum(B) - t1 * dotp(np.exp(- u / t1), r) - t2 * dotp(np.exp(- v / t2), c)

    return f


def unreg_f(B, C, r, c, t1, t2):
    """
    the unregularized objective with solutions u, v
    """
    a = B.sum(axis=1).reshape(-1, 1)
    b = B.sum(axis=0).reshape(-1, 1)
    return dotp(C, B) + t1 * get_KL(a, r) + t2 * get_KL(b, c)


def f_primal(unreg_f_val, B, eta):
    ent = get_entropy(B) 
    return unreg_f_val - eta * ent


def sinkhorn_uot(C, r, c, eta=1.0, t1=1.0, t2=1.0, n_iter=100, duals=False):
    """
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg eta: entropic-regularizer
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    :n_iter: number of Sinkhorn iterations
    """


    # initial solution
    u = np.zeros(r.shape)
    v = np.zeros(c.shape)


    for i in range(n_iter):
        u_old = copy(u)
        v_old = copy(v)
        # update
        if i % 2 == 0:
            a = B.sum(axis=1).reshape(-1, 1)
            u = (u / eta + np.log(r) - np.log(a)) * (t1 * eta / (eta + t1))
        else:
            b = B.sum(axis=0).reshape(-1, 1)
            v = (v / eta + np.log(c) - np.log(b)) * (t2 * eta / (eta + t2))


        err = norm1(u - u_old) + norm1(v - v_old)


        if err < 1e-10:
            stop_iter = i + 1
            break

    B = get_B(C, u, v, eta)

    if duals:
        return B, u, v
    else:
        return B



def fw_uot(C, r, c, t1=1.0, t2=1.0, n_iter=100):
    """
    frank wolfe for full uot. we take advantage of the upperbound on \sum_{ij} X_{ij}
    as shown in Pham et al. 2020
    """

    log_r = np.log(r)
    log_c = np.log(c).reshape(1, -1)

    # X = np.random.rand(*C.shape)
    mu = np.sqrt(np.sum(r) * np.sum(c))
    X = np.ones(C.shape) * mu / C.size

    
    for it in range(n_iter):
        tau = 2 / (it + 3)
        log_sum_r = np.log(X.sum(axis=-1, keepdims=True)) # [r_dim, 1]
        log_sum_c = np.log(X.sum(axis=0, keepdims=True)) # [1, c_dim]
        delta = C + t1 * log_sum_r + t2 * log_sum_c - t1 * log_r - t2 * log_c

        min_indx = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[min_indx] >= 0:
            X = (1 - tau) * X
        else:
            V = np.zeros(delta.shape)
            V[min_indx] = mu
            X = (1 - tau) * X + tau * V
        

    return X
    

