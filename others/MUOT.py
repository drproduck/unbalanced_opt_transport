import numpy as np
import matplotlib.pyplot as plt
from utils import *
from copy import copy
from time import time

def get_except(n, i):
    return tuple(range(i)) + tuple(range(i+1,n))

def get_B(C, U, eta):
    n = len(C.shape)

    K = - C
    for i in range(n):
        expand_dims = [1]*n
        expand_dims[i] = len(U[i])
        K += U[i].reshape(expand_dims)

    return np.exp(K / eta)

def f_dual(B, U, R, eta, T):
    """
    the dual of the entropic-regularized unbalanced OT
    """
    n = len(B.shape)
    transport = np.sum(B)
    f = eta * transport
    margs = []
    for i in range(n):
        margs.append(dotp(np.exp(- U[i] / T[i]), R[i]))
        f += T[i] * margs[i]
    return f, transport, margs


def f_primal(C, B, U, R, eta, T):
    """
    the primal objective with solutions B
    """
    n = len(B.shape)
    f = dotp(C, B)
    for i in range(n):
        ai = B.sum(axis=get_except(n, i)).reshape(-1, 1)
        f += T[i] * get_KL(ai, R[i])

    f -= eta * get_entropy(B)
    return f

def unreg_f(C, B, U, R, eta, T):
    """
    the unregularized objective with solutions u, v
    """
    n = len(B.shape)
    f = dotp(C, B)
    for i in range(n):
        ai = B.sum(axis=get_except(n, i)).reshape(-1, 1)
        f += T[i] * get_KL(ai, R[i])

    return f



def sinkhorn_muot(C, R, eta, T, n_iter=100, early_stop=True):
    """
    :arg C: cost matrix [d1 * d2 * ... * dn]
    :arg R: list of marginals
    :arg eta: entropic-regularizer
    :arg T: list of KL regularizer
    :n_iter: number of Sinkhorn iterations
    """
    dims = C.shape
    n = len(dims)

    # collect some stats
    f_primal_val_list = []
    f_dual_val_list = []
    unreg_f_val_list = []
    err_list = []
    time_per_iter_list = []

    # all_solutions
    U_all = []
    # initial solution
    U = []
    for d in dims:
        U.append(np.zeros(shape=(d, 1)))


    # compute before any updates
    B = get_B(C, U, eta)

    f_primal_val = f_primal(C, B, U, R, eta, T)
    f_primal_val_list.append(f_primal_val)
    f_dual_val = f_dual(B, U, R, eta, T)
    f_dual_val_list.append(f_dual_val)
    unreg_f_val = unreg_f(C, B, U, R, eta, T)
    unreg_f_val_list.append(unreg_f_val)


    stop_iter = n_iter
    for i in range(n_iter):
        sol_it = []

        j = i % n
        old = copy(U[j])

        # update
        start = time()
        aj = B.sum(axis=get_except(n, j)).reshape(-1, 1)
        U[j] = (U[j] / eta + np.log(R[j]) - np.log(aj)) * (T[j] * eta / (eta + T[j]))

        #save solution of this iter
        for k in range(n):
            sol_it.append(copy(U[k]))
        U_all.append(sol_it)

        B = get_B(C, U, eta)
        stop = time()

        # post-process
        time_per_iter = stop - start
        time_per_iter_list.append(time_per_iter)
        f_primal_val = f_primal(C, B, U, R, eta, T)
        f_primal_val_list.append(f_primal_val)
        f_dual_val, _, _ = f_dual(B, U, R, eta, T)
        f_dual_val_list.append(f_dual_val)
        unreg_f_val = unreg_f(C, B, U, R, eta, T)
        unreg_f_val_list.append(unreg_f_val)

        new = U[j]
        err = norm1(new - old)
        err_list.append(err)

#         if early_stop and err < 1e-10:
#             stop_iter = i + 1
#             break

    info = {}
    info['f_primal_val_list'] = f_primal_val_list
    info['f_dual_val_list'] = f_dual_val_list
    info['unreg_f_val_list'] = unreg_f_val_list
    info['err_list'] = err_list
    info['time_per_iter_list'] = time_per_iter_list
    if early_stop:
        info['stop_iter'] = stop_iter
    else: info['stop_iter'] = n_iter
    info['U_all'] = U_all

    return info
