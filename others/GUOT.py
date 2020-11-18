import numpy as np
import matplotlib.pyplot as plt
from utils import *
from copy import copy
import pdb

def B_coo_update(B, a, b, vec, coo, old, new, eta):
    r_size, c_size = B.shape
    if vec == 'u':
        b -= B[coo, :].reshape(c_size, 1)
        B[coo, :] *= np.exp((new - old) / eta)
        a[coo] = B[coo, :].sum()
        b += B[coo, :].reshape(c_size, 1)

    if vec == 'v':
        a -= B[:, coo].reshape(r_size, 1)
        B[:, coo] *= np.exp((new - old) / eta)
        b[coo] = B[:, coo].sum()
        a += B[:, coo].reshape(r_size, 1)



def f(B, u, v, r, c, eta, t1, t2):
    """
    the dual of the entropic-regularized unbalanced OT
    """
    f = eta * np.sum(B) + t1 * dotp(np.exp(- u / t1) - 1, r) + t2 * dotp(np.exp(- v / t2) - 1, c)

    return f

def unreg_f(B, C, u, v, r, c, eta, t1, t2):
    """
    the unregularized objective with solutions u, v
    """
    a = B.sum(axis=1).reshape(1, -1)
    b = B.sum(axis=0).reshape(1, -1)
    return dotp(B, C) + t1 * get_KL(a, r) + t2 * get_KL(b, c)



def sinkhorn_guot(C, r, c, eta=1.0, t1=1.0, t2=1.0, n_iter=100):
    """
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg eta: entropic-regularizer
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    :n_iter: number of Sinkhorn iterations
    """

    r_size = len(r)
    c_size = len(c)

    # will take square matrix for now
    assert(r_size == c_size)
    m = r_size + c_size

    # collect some stats
    f_val_list = []
    unreg_f_val_list = []
    err_list = []

    # initial solution
    u = np.zeros(r.shape)
    v = np.zeros(c.shape)

    B = np.exp((-C + u + v.T) / eta)
    a = B.sum(axis=1).reshape(r_size, 1)
    b = B.sum(axis=0).reshape(c_size, 1)

    # compute before any updates
    f_val = f(B, u, v, r, c, eta, t1, t2)
    f_val_list.append(f_val)
    unreg_f_val = unreg_f(B, C, u, v, r, c, eta, t1, t2)
    unreg_f_val_list.append(unreg_f_val)

    # stop_iter = n_iter

    for k in range(n_iter):

        # update
        for i in range(m):
            if k % m == i:
                if i <= m / 2 - 1:
                    j = i

                    u_old = u[j, 0]
                    u[j] = (u[j] / eta + np.log(r[j]) - np.log(a[j])) * (t1 * eta / (eta + t1))
                    u_new = u[j, 0]
                    B_coo_update(B, a, b, vec='u', coo=j, old=u_old, new=u_new, eta=eta)
                    

                else:
                    j = int(i - m / 2)

                    v_old = v[j, 0]
                    v[j] = (v[j] / eta + np.log(c[j]) - np.log(b[j])) * (t2 * eta / (eta + t2))
                    v_new = v[j, 0]
                    B_coo_update(B, a, b, vec='v', coo=j, old=v_old, new=v_new, eta=eta)



        f_val = f(B, u, v, r, c, eta, t1, t2)
        f_val_list.append(f_val)
        unreg_f_val = unreg_f(B, C, u, v, r, c, eta, t1, t2)
        unreg_f_val_list.append(unreg_f_val)

        # err = norm1(u - u_old) + norm1(v - v_old)
        # err_list.append(err)

        # if err < 1e-10:
        #     stop_iter = i + 1
        #     break

    info = {}
    info['f_val_list'] = f_val_list
    info['unreg_f_val_list'] = unreg_f_val_list
    # info['err_list'] = err_list
    # info['stop_iter'] = stop_iter

    return u, v, info
