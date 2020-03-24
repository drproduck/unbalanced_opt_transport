import numpy as np
from utils import *
from copy import copy
import numpy

def get_B(C, u, v, eta):
    n, m = C.shape
    K = - C + u + v.T
    return np.exp(K / eta)

def f_dual(B, u, v, r, c, eta, t1, t2):
    """
    the dual of the entropic-regularized unbalanced OT
    """
    f = eta * np.sum(B) + t1 * dotp(np.exp(- u / t1), r) + t2 * dotp(np.exp(- v / t2), c)

    return f


def unreg_f(B, C, r, c, eta, t1, t2):
    """
    the unregularized objective with solutions u, v
    """
    a = B.sum(axis=1).reshape(-1, 1)
    b = B.sum(axis=0).reshape(-1, 1)
    return dotp(C, B) + t1 * get_KL(a, r) + t2 * get_KL(b, c)


def f_primal(unreg_f_val, B, eta):
    ent = get_entropy(B) 
    return unreg_f_val - eta * ent


def sinkhorn_uot(C, r, c, eta=1.0, t1=1.0, t2=1.0, n_iter=100, early_stop=True, eps=None, opt_val=None, save_uv=True):
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
    f_primal_val_list = []
    sum_B_list = []
    err_list = []

    # initial solution
    u = np.zeros(r.shape).astype(numpy.longdouble)
    v = np.zeros(c.shape).astype(numpy.longdouble)

    if save_uv:
        u_list.append(u)
        v_list.append(v)

    # compute before any updates
    B = get_B(C, u, v, eta)
    f_val = f_dual(B, u, v, r, c, eta, t1, t2)
    f_val_list.append(f_val)

    unreg_f_val = unreg_f(B, C, r, c, eta, t1, t2)
    unreg_f_val_list.append(unreg_f_val)

    f_primal_val = f_primal(unreg_f_val, B, eta)
    f_primal_val_list.append(f_primal_val)

    sum_B_list.append(B.sum())



    stop_iter = n_iter
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

        if save_uv:
            u_list.append(copy(u))
            v_list.append(copy(v))
	
	# compute stats
        B = get_B(C, u, v, eta)

        f_val = f_dual(B, u, v, r, c, eta, t1, t2)
        f_val_list.append(f_val)

        unreg_f_val = unreg_f(B, C, r, c, eta, t1, t2)
        unreg_f_val_list.append(unreg_f_val)

        f_primal_val = f_primal(unreg_f_val, B, eta)
        f_primal_val_list.append(f_primal_val)

        sum_B_list.append(B.sum())

        err = norm1(u - u_old) + norm1(v - v_old)
        err_list.append(err)

        if eps is not None and unreg_f_val_list[-1] <= opt_val + eps:
            stop_iter = i + 1
            break

        if early_stop and err < 1e-10:
            stop_iter = i + 1
            break

    info = {}
    if save_uv:
        info['u_list'] = u_list
        info['v_list'] = v_list
    info['f_val_list'] = f_val_list
    info['unreg_f_val_list'] = unreg_f_val_list
    info['f_primal_val_list'] = f_primal_val_list
    info['sum_B_list'] = sum_B_list
    info['err_list'] = err_list
    if early_stop or eps is not None:
        info['stop_iter'] = stop_iter
    else: info['stop_iter'] = n_iter

    return u, v, info 
