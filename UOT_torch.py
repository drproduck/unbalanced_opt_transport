import numpy as np
from utils import *
import numpy
import torch


# utilities
def get_entropy(P):
    logP = torch.log(P + 1e-20)
    return -1 * torch.sum(logP * P - P)

def get_KL(P, Q):
    log_ratio = torch.log(P + 1e-20) - torch.log(Q + 1e-20)
    return torch.sum(P * log_ratio - P + Q)

def dotp(x, y):
    return torch.sum(x * y)

def norm1(X):
    return torch.sum(torch.abs(X))

def supnorm(X):
    return torch.max(torch.abs(X))


# main funcs
def get_B(C, u, v, eta):
    n, m = C.shape
    K = - C + u + v.T
    return torch.exp(K / eta)


def f_dual(B, u, v, r, c, eta, t1, t2):
    """
    the dual of the entropic-regularized unbalanced OT
    """
    f = eta * torch.sum(B) + t1 * dotp(torch.exp(- u / t1), r) + t2 * dotp(torch.exp(- v / t2), c)

    return f


def unreg_f(B, C, r, c, eta, t1, t2):
    """
    the unregularized objective with solutions u, v
    """
    a = B.sum(dim=1).reshape(-1, 1)
    b = B.sum(dim=0).reshape(-1, 1)
    return dotp(C, B) + t1 * get_KL(a, r) + t2 * get_KL(b, c)


def f_primal(unreg_f_val, B, eta):
    ent = get_entropy(B) 
    return unreg_f_val - eta * ent



def sinkhorn_uot(C, r, c, eta=1.0, t1=1.0, t2=1.0, n_iter=100, early_stop=True, eps=None, opt_val=None):
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
    f_val_list = []
    unreg_f_val_list = []
    f_primal_val_list = []
    sum_B_list = []

    # initial solution
    u = torch.zeros(r.shape, dtype=torch.float64)
    v = torch.zeros(c.shape, dtype=torch.float64)

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
        # update
        if i % 2 == 0:
            a = B.sum(dim=1).reshape(-1, 1)
            u = (u / eta + torch.log(r) - torch.log(a)) * (t1 * eta / (eta + t1))
        else:
            b = B.sum(dim=0).reshape(-1, 1)
            v = (v / eta + torch.log(c) - torch.log(b)) * (t2 * eta / (eta + t2))

	# compute stats
        B = get_B(C, u, v, eta)

        f_val = f_dual(B, u, v, r, c, eta, t1, t2)
        f_val_list.append(f_val)

        unreg_f_val = unreg_f(B, C, r, c, eta, t1, t2)
        unreg_f_val_list.append(unreg_f_val)

        f_primal_val = f_primal(unreg_f_val, B, eta)
        f_primal_val_list.append(f_primal_val)

        sum_B_list.append(B.sum())


        if eps is not None and unreg_f_val_list[-1] <= opt_val + eps:
            stop_iter = i + 1
            break


    info = {}
    info['f_val_list'] = f_val_list
    info['unreg_f_val_list'] = unreg_f_val_list
    info['f_primal_val_list'] = f_primal_val_list
    info['sum_B_list'] = sum_B_list
    info['stop_iter'] = stop_iter

    return u, v, info
