import numpy as np
import torch
from time import time


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


def unreg_f(B, C, r, c, t1, t2):
    """
    the unregularized objective with solutions u, v
    """
    a = B.sum(dim=1).reshape(-1, 1)
    b = B.sum(dim=0).reshape(-1, 1)
    return dotp(C, B) + t1 * get_KL(a, r) + t2 * get_KL(b, c)


def f_primal(unreg_f_val, B, eta):
    ent = get_entropy(B) 
    return unreg_f_val - eta * ent



def sinkhorn_uot(C, r, c, eta=1.0, t1=1.0, t2=1.0, n_iter=100, eps=None, opt_val=None, vbo=False):
    """
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg eta: entropic-regularizer
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    :n_iter: number of Sinkhorn iterations
    """

    # convert numpy variables to torch
    C = torch.from_numpy(C)
    r = torch.from_numpy(r)
    c = torch.from_numpy(c)

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

    unreg_f_val = unreg_f(B, C, r, c, t1, t2)
    unreg_f_val_list.append(unreg_f_val)

    f_primal_val = f_primal(unreg_f_val, B, eta)
    f_primal_val_list.append(f_primal_val)

    sum_B_list.append(B.sum())


    stop_iter = n_iter
    start = time()
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

        unreg_f_val = unreg_f(B, C, r, c, t1, t2)
        unreg_f_val_list.append(unreg_f_val)

        f_primal_val = f_primal(unreg_f_val, B, eta)
        f_primal_val_list.append(f_primal_val)

        sum_B_list.append(B.sum())


        if eps is not None and unreg_f_val_list[-1] <= opt_val + eps:
            stop_iter = i + 1
            break

        if vbo and (i + 1) % 1000 == 0:
            stop = time()
            print(f'iteration={i+1}, elapsed={stop-start:.3f}, f_dual={f_val:.3f}, f_primal={f_primal_val:.3f}, f_unreg={unreg_f_val:.3f}')
            start = time()


    info = {}
    info['f_val_list'] = f_val_list
    info['unreg_f_val_list'] = unreg_f_val_list
    info['f_primal_val_list'] = f_primal_val_list
    info['sum_B_list'] = sum_B_list
    info['stop_iter'] = stop_iter

    return u, v, info


def grad_descent_exp_uot(C, r, c, eta=1.0, t1=1.0, t2=1.0, gamma=0.01, n_iter=100):

    """
    :arg C: cost matrix shape = [r_dim, c_dim]
    :arg r: first marginal shape = [r_dim, 1]
    :arg c: second marginal shape = [c_dim, 1]
    :arg eta: entropic-regularizer
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    :arg gamma: step size
    :n_iter: number of Sinkhorn iterations
    """

    C = torch.from_numpy(C).type(torch.float32).cuda()
    r = torch.from_numpy(r).type(torch.float32).cuda()
    c = torch.from_numpy(c).type(torch.float32).cuda()
    
    X_list = []
    unreg_f_val_list = []
    f_primal_val_list = []
    grad_norm_list = []

    log_r = torch.log(r)
    log_c = torch.log(c).reshape(1, -1)

    # log_X = - C / eta
    log_X = torch.randn_like(C).cuda()
    log_X.requires_grad = True
    X = torch.exp(log_X)

    optimizer = torch.optim.Adam([log_X], lr=gamma)

    
    for it in range(n_iter):
        unreg_f_val = unreg_f(X, C, r, c, t1, t2)
        f_primal_val = f_primal(unreg_f_val, X, eta)
        optimizer.zero_grad()
        f_primal_val.backward()
        optimizer.step()

        X = torch.exp(log_X)
        grad_norm = log_X.grad.data.norm(2).item()

        X_list.append(X.detach().cpu().numpy())
        unreg_f_val_list.append(unreg_f_val.item())
        f_primal_val_list.append(f_primal_val.item())
        grad_norm_list.append(grad_norm)


    info = {'X_list': X_list,
            'unreg_f_val_list': unreg_f_val_list,
            'f_primal_val_list': f_primal_val_list,
            'grad_norm_list': grad_norm_list,
            }

    return X.detach().cpu().numpy(), info
