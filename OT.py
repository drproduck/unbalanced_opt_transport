import numpy as np
import matplotlib.pyplot as plt
from utils import *

def get_B(C, u, v, eta):
    return np.exp(- C / eta + u + v.T)

def f(C, u, v, r, c, eta):
    """
    the dual of the entropic-regularized balanced OT
    """
    A = get_B(C, u, v, eta)
    f = np.sum(A) + dotp(u, r) + dotp(v, c)
    return f

def f_primal(C, u, v, eta):
    A = get_B(C, u, v, eta)
    unreg_val = dotp(C, A)

    entropy = get_entropy(A)

    return unreg_val - eta * entropy

def unreg_f(C, u, v, eta):
    """
    the unregularized objective with solutions u, v
    """
    A = get_B(C, u, v, eta)

    return dotp(C, A)

def norm1_constraint(C, u, v, r, c, eta):
    A = get_B(C, u, v, eta)
    a = A.sum(axis=1).reshape(-1, 1)
    b = A.sum(axis=0).reshape(-1, 1)
    return norm1(a - r) + norm1(b - c)


def sinkhorn_ot(C, r, c, eta=1.0, n_iter=1000):
    """
    log stable balanced OT
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    """

    # collect some stats
    u_list = []
    v_list = []
    f_val_list = []
    f_primal_val_list = []
    unreg_f_val_list = []
    constraint_norm_list = []

    # initial solutions
    u = np.zeros(r.shape)
    v = np.zeros(c.shape)

    u_list.append(u)
    v_list.append(v)

    # compute before any updates
    f_val = f(C, u, v, r, c, eta)
    f_val_list.append(f_val)

    unreg_f_val = unreg_f(C, u, v, eta)
    unreg_f_val_list.append(unreg_f_val)

    f_primal_val = f_primal(C, u, v, eta)
    f_primal_val_list.append(f_primal_val)

    rc_diff = norm1_constraint(C, u, v, r, c, eta)
    constraint_norm_list.append(rc_diff)

    stop_iter = n_iter
    for i in range(n_iter):
        A = get_B(C, u, v, eta)
        if i % 2 == 0:
            a = A.sum(axis=1).reshape(-1, 1)
            u = u + np.log(r) - np.log(a)
        else:
            b = A.sum(axis=0).reshape(-1, 1)
            v = v + np.log(c) - np.log(b)

        u_list.append(u)
        v_list.append(v)

        f_val = f(C, u, v, r, c, eta)
        f_val_list.append(f_val)

        unreg_f_val = unreg_f(C, u, v, eta)
        unreg_f_val_list.append(unreg_f_val)

        f_primal_val = f_primal(C, u, v, eta)
        f_primal_val_list.append(f_primal_val)

        rc_diff = norm1_constraint(C, u, v, r, c, eta)
        constraint_norm_list.append(rc_diff)

        if rc_diff < 1e-10:
            stop_iter = i + 1 
            break

    info = {} 
    info['u_list'] = u_list
    info['v_list'] = v_list
    info['f_val_list'] = f_val_list
    info['unreg_f_val_list'] = unreg_f_val_list
    info['f_primal_val_list'] = f_primal_val_list
    info['constraint_norm_list'] = constraint_norm_list

    info['stop_iter'] = stop_iter

    return u_list[-1], v_list[-1], info
