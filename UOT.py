import numpy as np
import matplotlib.pyplot as plt
from utils import *
from copy import copy

def get_B(C, u, v, eta):
    n, m = C.shape
    K = - C + u + v.T
    return np.exp(K / eta)

def f(C, u, v, r, c, eta, t1, t2):
    """
    the dual of the entropic-regularized unbalanced OT
    """
    A = get_B(C, u, v, eta)
    f = eta * np.sum(A) + t1 * dotp(np.exp(- u / t1) - 1, r) + t2 * dotp(np.exp(- v / t2) - 1, c)

    return f

def unreg_f(C, u, v, r, c, eta, t1, t2):
    """
    the unregularized objective with solutions u, v
    """
    A = get_B(C, u, v, eta)
    a = A.sum(axis=1).reshape(1, -1)
    b = A.sum(axis=0).reshape(1, -1)
    return dotp(C, A) + t1 * get_KL(a, r) + t2 * get_KL(b, c)



def sinkhorn_uot(C, r, c, eta=1.0, t1=1.0, t2=1.0, n_iter=100, early_stop=True):
    """
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg eta: entropic-regularizer
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    :n_iter: number of Sinkhorn iterations
    """

    # collect some stats
    u_list = []
    v_list = []
    f_val_list = []
    unreg_f_val_list = []
    err_list = []

    # initial solution
    u = np.zeros(r.shape)
    v = np.zeros(c.shape)

    u_list.append(u)
    v_list.append(v)

    # compute before any updates
    f_val = f(C, u, v, r, c, eta, t1, t2)
    f_val_list.append(f_val)
    unreg_f_val = unreg_f(C, u, v, r, c, eta, t1, t2)
    unreg_f_val_list.append(unreg_f_val)


    stop_iter = n_iter
    for i in range(n_iter):
        u_old = copy(u)
        v_old = copy(v)
        A = get_B(C, u, v, eta)
        # update
        if i % 2 == 0:
            a = A.sum(axis=1).reshape(-1, 1)
            u = (u / eta + np.log(r) - np.log(a)) * (t1 * eta / (eta + t1))
        else:
            b = A.sum(axis=0).reshape(-1, 1)
            v = (v / eta + np.log(c) - np.log(b)) * (t2 * eta / (eta + t2))

        u_list.append(u)
        v_list.append(v)

        f_val = f(C, u, v, r, c, eta, t1, t2)
        f_val_list.append(f_val)
        unreg_f_val = unreg_f(C, u, v, r, c, eta, t1, t2)
        unreg_f_val_list.append(unreg_f_val)

        err = norm1(u - u_old) + norm1(v - v_old)
        err_list.append(err)

        if early_stop and err < 1e-10:
            stop_iter = i + 1
            break

    info = {}
    info['u_list'] = u_list
    info['v_list'] = v_list
    info['f_val_list'] = f_val_list
    info['unreg_f_val_list'] = unreg_f_val_list
    info['err_list'] = err_list
    if early_stop:
        info['stop_iter'] = stop_iter
    else: info['stop_iter'] = n_iter

    return u_list[-1], v_list[-1], info
