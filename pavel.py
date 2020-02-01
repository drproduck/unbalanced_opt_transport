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

def find_rc(r, c, epsp, n):
    """check line 2 of algorithm 2"""

    r = (1 - epsp / 8) * (r + epsp / n / (8 - epsp))
    c = (1 - epsp / 8) * (c + epsp / n / (8 - epsp))
    return r, c



def sinkhorn_ot(C, r, c, eps, opt_val):
    """
    log stable balanced OT
    Note we run sinkhorn with rt, ct as in the paper. From here we compute the regularized obj with rt,ct but the unregularized with r, c with projection
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    """
    n = C.shape[0]
    maxC = C.max()
    eta = eps / (4 * np.log(n))
    epsp = eps / (8 * maxC)
    rt, ct = find_rc(r, c, epsp, n)

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
    A = get_B(C, u, v, eta)

    f_val = f(C, u, v, rt, ct, eta)
    f_val_list.append(f_val)

    unreg_f_val = dotp(C, round_rc(A, r, c))
    unreg_f_val_list.append(unreg_f_val)

    f_primal_val = f_primal(C, u, v, eta)
    f_primal_val_list.append(f_primal_val)

    k = 0
    opt_iter = None
    while norm1(A.sum(axis=1).reshape(-1, 1) - rt) + norm1(A.sum(axis=0).reshape(-1, 1) - ct) > epsp / 2:
        k = k + 1
        if k % 2 == 1:
            a = A.sum(axis=1).reshape(-1, 1)
            u = u + np.log(rt) - np.log(a)
        else:
            b = A.sum(axis=0).reshape(-1, 1)
            v = v + np.log(ct) - np.log(b)

        A = get_B(C, u, v, eta)
        u_list.append(u)
        v_list.append(v)

        f_val = f(C, u, v, rt, ct, eta)
        f_val_list.append(f_val)

        unreg_f_val = dotp(C, round_rc(A, r, c))
        unreg_f_val_list.append(unreg_f_val)

        if unreg_f_val < opt_val + eps and opt_iter is None:
            print(unreg_f_val, opt_val, eps)
            opt_iter = k

        f_primal_val = f_primal(C, u, v, eta)
        f_primal_val_list.append(f_primal_val)


    info = {} 
    info['u_list'] = u_list
    info['v_list'] = v_list
    info['f_val_list'] = f_val_list
    info['unreg_f_val_list'] = unreg_f_val_list
    info['f_primal_val_list'] = f_primal_val_list

    info['stop_iter'] = k
    info['opt_iter'] = opt_iter

    return u_list[-1], v_list[-1], info
