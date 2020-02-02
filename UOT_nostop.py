import numpy as np
from utils import *
from copy import copy
import numpy

def get_B(C, u, v, eta):
    n, m = C.shape
    K = - C + u + v.T
    return np.exp(K / eta)

def f(C, u, v, r, c, eta, t1, t2):
    """
    the dual of the entropic-regularized unbalanced OT
    """
    A = get_B(C, u, v, eta)
    f = eta * np.sum(A) + t1 * dotp(np.exp(- u / t1), r) + t2 * dotp(np.exp(- v / t2), c)

    return f

def f_primal(C, u, v, r, c, eta, t1, t2):
    A = get_B(C, u, v, eta)
    a = A.sum(axis=1).reshape(-1, 1)
    b = A.sum(axis=0).reshape(-1, 1)
    unreg_f_val = dotp(C, A) + t1 * get_KL(a, r) + t2 * get_KL(b, c)
    ent = get_entropy(A) 
    return unreg_f_val - eta * ent


def unreg_f(C, u, v, r, c, eta, t1, t2):
    """
    the unregularized objective with solutions u, v
    """
    A = get_B(C, u, v, eta)
    a = A.sum(axis=1).reshape(-1, 1)
    b = A.sum(axis=0).reshape(-1, 1)
    return dotp(C, A) + t1 * get_KL(a, r) + t2 * get_KL(b, c)



def sinkhorn_uot(C, r, c, eta=1.0, t1=1.0, t2=1.0, eps=None, opt_val=None):
    """
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg eta: entropic-regularizer
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    """

    # collect some stats
    u_list = []
    v_list = []
    f_val_list = []
    f_primal_val_list = []
    unreg_f_val_list = []
    sum_P_list = []
    entropy_list = []
    kl_list = []
    err_list = []

    # initial solution
    u = np.zeros(r.shape).astype(numpy.longdouble)
    v = np.zeros(c.shape).astype(numpy.longdouble)

    u_list.append(u)
    v_list.append(v)

    # compute before any updates
    f_val = f(C, u, v, r, c, eta, t1, t2)
    f_val_list.append(f_val)

    f_primal_val = f_primal(C, u, v, r, c, eta, t1, t2)
    f_primal_val_list.append(f_primal_val)

    unreg_f_val = unreg_f(C, u, v, r, c, eta, t1, t2)
    unreg_f_val_list.append(unreg_f_val)

    P = get_B(C, u, v, eta)
    sum_P_list.append(P.sum())

    entropy = eta * get_entropy(P)
    entropy_list.append(entropy)

    a = P.sum(axis=1).reshape(-1, 1)
    b = P.sum(axis=0).reshape(-1, 1)
    kl = t1 * get_KL(a, r) + t2 * get_KL(b, c)
    kl_list.append(kl)


    k = 0
    while True:
        k = k + 1
        u_old = copy(u)
        v_old = copy(v)
        A = get_B(C, u, v, eta)
        # update
        if k % 2 == 1:
            a = A.sum(axis=1).reshape(-1, 1)
            u = (u / eta + np.log(r) - np.log(a)) * (t1 * eta / (eta + t1))
        else:
            b = A.sum(axis=0).reshape(-1, 1)
            v = (v / eta + np.log(c) - np.log(b)) * (t2 * eta / (eta + t2))

        u_list.append(copy(u))
        v_list.append(copy(v))

        f_val = f(C, u, v, r, c, eta, t1, t2)
        f_val_list.append(f_val)

        f_primal_val = f_primal(C, u, v, r, c, eta, t1, t2)
        f_primal_val_list.append(f_primal_val)

        unreg_f_val = unreg_f(C, u, v, r, c, eta, t1, t2)
        unreg_f_val_list.append(unreg_f_val)

        P = get_B(C, u, v, eta)
        sum_P_list.append(P.sum())

        entropy = eta * get_entropy(P)
        entropy_list.append(entropy)

        a = P.sum(axis=1).reshape(-1, 1)
        b = P.sum(axis=0).reshape(-1, 1)
        kl = t1 * get_KL(a, r) + t2 * get_KL(b, c)
        kl_list.append(kl)

        err = norm1(u - u_old) + norm1(v - v_old)
        err_list.append(err)

        if unreg_f_val_list[-1] <= opt_val + eps:
            break


    info = {}
    info['u_list'] = u_list
    info['v_list'] = v_list
    info['f_val_list'] = f_val_list
    info['f_primal_val_list'] = f_primal_val_list
    info['unreg_f_val_list'] = unreg_f_val_list
    info['sum_P_list'] = sum_P_list
    info['entropy_list'] = entropy_list
    info['kl_list'] = kl_list
    info['err_list'] = err_list

    info['stop_iter'] = k

    return u_list[-1], v_list[-1], info