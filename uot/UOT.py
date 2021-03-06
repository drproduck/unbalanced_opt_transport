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

    unreg_f_val = unreg_f(B, C, r, c, t1, t2)
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

        unreg_f_val = unreg_f(B, C, r, c, t1, t2)
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

    return B, info 


def grad_descent_uot(C, r, c, eta=1.0, t1=1.0, t2=1.0, gamma=0.01, n_iter=100):

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
    
    X_list = []
    unreg_f_val_list = []
    f_primal_val_list = []
    grad_norm_list = []

    log_r = np.log(r)
    log_c = np.log(c).reshape(1, -1)

    X = np.random.rand(*C.shape)
    unreg_f_val = unreg_f(X, C, r, c, t1, t2)
    f_primal_val = f_primal(unreg_f_val, X, eta)

    X_list.append(X)
    unreg_f_val_list.append(unreg_f_val)
    f_primal_val_list.append(f_primal_val)
    
    for it in range(n_iter):
        log_sum_r = np.log(X.sum(axis=-1, keepdims=True)) # [r_dim, 1]
        log_sum_c = np.log(X.sum(axis=0, keepdims=True)) # [1, c_dim]
        delta = C + t1 * log_sum_r + t2 * log_sum_c - t1 * log_r - t2 * log_c + eta * np.log(X)

        delta[~ np.isfinite(delta)] = 0
        
        X = X - gamma * delta
        X[X < 0] = 0

        unreg_f_val = unreg_f(X, C, r, c, t1, t2)
        f_primal_val = f_primal(unreg_f_val, X, eta)

        X_list.append(X)
        unreg_f_val_list.append(unreg_f_val)
        f_primal_val_list.append(f_primal_val)
        grad_norm_list.append(normfro(delta))

    info = {'X_list': X_list,
            'unreg_f_val_list': unreg_f_val_list,
            'f_primal_val_list': f_primal_val_list,
            'grad_norm_list': grad_norm_list,
            }

    return X, info


def backtracking_linesearch(eval_func, x, delta, alpha, beta):
    f = eval_func(x)
    x_hat = x - alpha * delta
    f_hat = eval_func(x_hat)
    # print(f_hat, f)
    while np.sum(x_hat < 0) > 0 or not f_hat <= f - alpha * beta * np.sum(delta**2):
        alpha /= 2
        if alpha < 1e-10: break
        x_hat = x - alpha * delta
        f_hat = eval_func(x_hat)
        # print(f_hat, f)
    return f_hat, x_hat, alpha


def grad_descent_uot_with_linesearch(C, r, c, eta=1.0, t1=1.0, t2=1.0, n_iter=100, alpha=0.1, beta=0.1):

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


    unreg_f_val_func = partial(unreg_f, C=C, r=r, c=c, eta=eta, t1=t1, t2=t2)

    
    X_list = []
    unreg_f_val_list = []
    f_primal_val_list = []
    grad_norm_list = []

    log_r = np.log(r)
    log_c = np.log(c).reshape(1, -1)

    # X = np.exp(- C / eta)
    X = np.random.rand(*C.shape)
    unreg_f_val = unreg_f(X, C, r, c, t1, t2)
    f_primal_val = f_primal(unreg_f_val, X, eta)

    X_list.append(X)
    unreg_f_val_list.append(unreg_f_val)
    f_primal_val_list.append(f_primal_val)

    
    for it in range(n_iter):
        log_sum_r = np.log(X.sum(axis=-1, keepdims=True)) # [r_dim, 1]
        log_sum_c = np.log(X.sum(axis=0, keepdims=True)) # [1, c_dim]
        delta = C + t1 * log_sum_r + t2 * log_sum_c - t1 * log_r - t2 * log_c + eta * np.log(X)
        
        unreg_f_val, X, alpha_hat = backtracking_linesearch(unreg_f_val_func, X, delta, alpha, beta)
        f_primal_val = f_primal(unreg_f_val, X, eta)

        X_list.append(X)
        unreg_f_val_list.append(unreg_f_val)
        f_primal_val_list.append(f_primal_val)
        grad_norm_list.append(normfro(delta))

    info = {'X_list': X_list,
            'unreg_f_val_list': unreg_f_val_list,
            'f_primal_val_list': f_primal_val_list,
            'grad_norm_list': grad_norm_list,
            }

    return X, info
    
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
    
    X_list = []
    unreg_f_val_list = []
    f_primal_val_list = []
    grad_norm_list = []

    log_r = np.log(r)
    log_c = np.log(c).reshape(1, -1)

    # log_X = - C / eta
    log_X = np.random.randn(*C.shape)
    X = np.exp(log_X)
    unreg_f_val = unreg_f(X, C, r, c, t1, t2)
    f_primal_val = f_primal(unreg_f_val, X, eta)

    X_list.append(X)
    unreg_f_val_list.append(unreg_f_val)
    f_primal_val_list.append(f_primal_val)
    
    for it in range(n_iter):
        lse_r = logsumexp(log_X, axis=-1, keepdims=True) # [r_dim, 1]
        lse_c = logsumexp(log_X, axis=0, keepdims=True) # [1, c_dim]
        dklr = (lse_r - log_r)
        dklc = (lse_c - log_c)
        dh = log_X
        delta = (C + t1 * dklr + t2 * dklc + eta * dh) * X
        log_X = log_X - gamma * delta 

        X = np.exp(log_X)
        unreg_f_val = unreg_f(X, C, r, c, t1, t2)
        f_primal_val = f_primal(unreg_f_val, X, eta)

        X_list.append(X)
        unreg_f_val_list.append(unreg_f_val)
        f_primal_val_list.append(f_primal_val)
        grad_norm_list.append(normfro(delta))

    info = {'X_list': X_list,
            'unreg_f_val_list': unreg_f_val_list,
            'f_primal_val_list': f_primal_val_list,
            'grad_norm_list': grad_norm_list,
            }

    return X, info


    
def grad_descent_unregularized_uot(C, r, c, t1=1.0, t2=1.0, n_iter=100, alpha=0.1, beta=0.5, subset=None, linesearch=False):

    """
    :arg C: cost matrix shape = [r_dim, c_dim]
    :arg r: first marginal shape = [r_dim, 1]
    :arg c: second marginal shape = [c_dim, 1]
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    :arg alpha: (initial) step size. Should set high for linesearch, low for fixed
    :arg beta: see armijo rule
    :arg n_iter: number of Sinkhorn iterations
    :arg linesearch:
    :arg subset: the subset of X to update
    """

    def update_subset(X, detla, subset):
        if subset is None:
            X_hat = X - alpha * delta
        else:
            X_hat = np.zeros_like(C)
            X_hat[subset] = X[subset] - alpha * delta[subset]
        X_hat[X_hat < 0] = 1e-100

        return X_hat

    X_list = []
    unreg_f_val_list = []
    f_primal_val_list = []
    grad_norm_list = []

    log_r = np.log(r)
    log_c = np.log(c).reshape(1, -1)

    # X = np.random.rand(*C.shape)
    X = np.exp(- C)
    unreg_f_val = unreg_f(X, C, r, c, t1, t2)

    X_list.append(X)
    unreg_f_val_list.append(unreg_f_val)
    
    for it in range(n_iter):
        log_sum_r = np.log(X.sum(axis=-1, keepdims=True)) # [r_dim, 1]
        log_sum_c = np.log(X.sum(axis=0, keepdims=True)) # [1, c_dim]
        delta = C + t1 * log_sum_r + t2 * log_sum_c - t1 * log_r - t2 * log_c

        if linesearch:
            # backtracking line search
            f = unreg_f_val_list[-1]
            X_hat = update_subset(X, delta, subset)
            f_hat = unreg_f(X_hat, C, r, c, t1, t2)
            while (not np.isfinite(f_hat)) or (not f_hat <= f - alpha * beta * np.sum(delta**2)):
                alpha /= 2
                X_hat = update_subset(X, delta, subset)
                f_hat = unreg_f(X_hat, C, r, c, t1, t2)

            X = X_hat
        else:
            X = update_subset(X, delta, subset)

        unreg_f_val = unreg_f(X, C, r, c, t1, t2)

        X_list.append(X)
        unreg_f_val_list.append(unreg_f_val)
        grad_norm_list.append(normfro(delta))

    info = {'X_list': X_list,
            'unreg_f_val_list': unreg_f_val_list,
            'grad_norm_list': grad_norm_list,
            }

    return X, info


def sigmoid(phi):
    return 1. / (1 + np.exp(- phi))


def nesterov_unregularized_uot(C, r, c, t1=1.0, t2=1.0, n_iter=100, lr=0.1, subset=None):

    """
    :arg C: cost matrix shape = [r_dim, c_dim]
    :arg r: first marginal shape = [r_dim, 1]
    :arg c: second marginal shape = [c_dim, 1]
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    :arg lr: step size.
    :arg n_iter: number of Sinkhorn iterations
    :arg linesearch:
    :arg subset: the subset of X to update
    """

    updater = NesterovGradDescent(lr)

    def update_subset(X, delta, subset):
        if subset is None:
            X_hat = updater.update(X, delta)
            X_hat[X_hat < 0] = 1e-100
        else:
            X_hat_subset = updater.update(X[subset], delta[subset])
            X_hat_subset[X_hat_subset < 0] = 1e-100
            X_hat = np.zeros_like(C)
            X_hat[subset] = X_hat_subset

        return X_hat

    X_list = []
    unreg_f_val_list = []
    f_primal_val_list = []
    grad_norm_list = []

    log_r = np.log(r)
    log_c = np.log(c).reshape(1, -1)

    X = np.random.rand(*C.shape)
    unreg_f_val = unreg_f(X, C, r, c, t1, t2)

    X_list.append(X)
    unreg_f_val_list.append(unreg_f_val)
    
    for it in range(n_iter):
        log_sum_r = np.log(X.sum(axis=-1, keepdims=True)) # [r_dim, 1]
        log_sum_c = np.log(X.sum(axis=0, keepdims=True)) # [1, c_dim]
        delta = C + t1 * log_sum_r + t2 * log_sum_c - t1 * log_r - t2 * log_c
        
        #TODO missing update here

        unreg_f_val = unreg_f(X, C, r, c, t1, t2)

        X_list.append(X)
        unreg_f_val_list.append(unreg_f_val)
        grad_norm_list.append(normfro(delta))

    info = {'X_list': X_list,
            'unreg_f_val_list': unreg_f_val_list,
            'grad_norm_list': grad_norm_list,
            }

    return X, info



def conditional_uot(C, r, c, t1=1.0, t2=1.0, n_iter=100):

    def step_v1(X_k):

    X_list = []
    unreg_f_val_list = []
    grad_norm_list = []

    log_r = np.log(r)
    log_c = np.log(c).reshape(1, -1)

    # X = np.random.rand(*C.shape)
    mu = np.sqrt(np.sum(r) * np.sum(c))
    X = np.ones(C.shape) * mu / C.size

    unreg_f_val = unreg_f(X, C, r, c, t1, t2)

    X_list.append(X)
    unreg_f_val_list.append(unreg_f_val)

    
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
        
        unreg_f_val = unreg_f(X, C, r, c, t1, t2)

        X_list.append(X)
        unreg_f_val_list.append(unreg_f_val)
        grad_norm_list.append(normfro(delta))

    info = {'X_list': X_list,
            'unreg_f_val_list': unreg_f_val_list,
            'grad_norm_list': grad_norm_list,
            }

    return X, info
    


def sparse_uot(C, r, c, eta=0.01, t1=1.0, t2=1.0, gamma=1., alpha=1e-3, n_sample=1, n_iter=100, subprob_args=None, vb=True):

    if subprob_args is None:
        subprob_args = {'alpha':1e-3, 'beta':1e-6, 'n_iter':1000, 'linesearch': False}

    grad_norm_list = []
    rand_f_list = []
    phi_list = []

    phi = np.random.randn(*C.shape)
    for it in range(n_iter):
        delta = 0.
        rand_f = 0
        for _ in range(n_sample):
            u = np.random.rand(*C.shape) # uniform
            index_more = (u > sigmoid(-phi))
            index_less  = (u < sigmoid(phi))
            f_more = grad_descent_unregularized_uot(C, r, c, t1, t2, **subprob_args, subset=index_more)[1]['unreg_f_val_list'][-1] + gamma * np.sum(index_more)
            f_less = grad_descent_unregularized_uot(C, r, c, t1, t2, **subprob_args, subset=index_less)[1]['unreg_f_val_list'][-1] + gamma * np.sum(index_less)
            # delta += (f_more - f_less) * (u - 0.5)
            delta += 0.5 * (f_more - f_less) * sigmoid(np.abs(phi)) * (index_more.astype(np.float) - index_less.astype(np.float))

            rand_f += f_less

        delta = delta / n_sample
        rand_f = rand_f / n_sample

        phi = phi - alpha * delta

        grad_norm = np.sqrt(np.sum(delta**2))

        grad_norm_list.append(grad_norm)
        rand_f_list.append(rand_f)
        phi_list.append(sigmoid(phi))

        if vb and it % 10 == 0:
            print(f'it {it}, grad_norm={grad_norm:.3f}')



    info = {'grad_norm_list': grad_norm_list,
            'rand_f_list': rand_f_list,
            'phi_list': phi_list,
            }

    return sigmoid(phi), info
