import numpy as np
import matplotlib.pyplot as plt
import pdb
import warnings
from copy import copy

np.seterr(all='warn')

# hyper parameters
t1 = 1.0
t2 = 1.0
eta = 1.0

rep_str = ''
rep_str += f'tau_1={t1:.3f}, tau_2={t2:.3f}, eta={eta:.3f}\n'

# def softmin(X, axis, eta):


def get_entropy(P):
    logP = np.log(P + 1e-20)
    return -1 * np.sum(logP * P - P)

def get_KL(P, Q):
    log_ratio = np.log(P) - np.log(Q)
    return  np.sum(P * log_ratio - P + Q)


def get_kl(B, a, axis):
    xB = B.sum(axis=axis).reshape(-1, 1)
    log_ratio = np.log(xB / a + 1e-20)
    ret = xB * log_ratio - xB + a
    return ret.sum()

def get_B(u, v, C):
    n, m = C.shape
    K = -C + u @ np.ones((1, m)) + np.ones((n, 1)) @ v.T
    return np.exp(K / eta)

def dotp(x, y):
    return np.sum(x * y)

def f(u, v, C, r, c):
    B = get_B(u, v, C)
    f = eta * np.sum(B) + t1 * dotp(np.exp(- u / t1) - 1, r) + t2 * dotp(np.exp(- v / t2) - 1, c)

    return f

def get_grad_norm(u, v, C, r, c, squared=True):
    B = get_B(u, v, C)
    rB = B.sum(axis=1).reshape(-1, 1)
    ur = np.exp(- u / t1) * r
    Du = rB - ur
    Du = np.abs(Du).sum().squeeze()

    cB = B.sum(axis=0).reshape(-1, 1)
    vc = np.exp(- v / t2) * c
    Dv = cB - vc
    Dv = np.abs(Dv).sum().squeeze()

    if squared:
        return Du**2 + Dv**2
    else:
        return Du + Dv

def norm1(x, y):
    return np.sum(np.abs(x - y))

# for n_pts in np.arange(1, 100):
eta_list = np.linspace(0.05, 1.0, 100)
eta_ratio_list = []
for eta in eta_list:
    # update values
    grad_norm_list = []
    f_val_list = []
    log_ratio_list = []
    unreg_obj_list = []
    reg_obj_list = []

    for n_pts in [10]:

        col_ones = np.ones((n_pts, 1))

        # problem-specific parameters (random)
        r = np.random.uniform(size=(n_pts, 1))
        # r = np.array([[1],[3]])
        rep_str += f'a={r}\n'
        c = np.random.uniform(size=(n_pts, 1))
        # c = np.array([[2],[4]])
        rep_str += f'b={c}\n'
        C = np.random.uniform(low=1, high=10, size=(n_pts, n_pts))
        C = (C + C.T) / 2
        # C = np.array([[8, 6],[6, 8]])
        rep_str += f'C={C}\n'
        u = np.zeros((n_pts, 1))
        v = np.zeros((n_pts, 1))


        # compute before any updates
        f_val = f(u, v, C, r, c)
        grad_norm = get_grad_norm(u, v, C, r, c, squared=False)
        f_val_list.append(f_val)
        grad_norm_list.append(grad_norm)

        n_iter = 1000
        for i in range(n_iter):

            B = get_B(u, v, C)
            a = B.sum(axis=1).reshape(-1, 1)
            b = B.sum(axis=0).reshape(-1, 1)

            # update
            if i % 2 == 0:
                u = (u / eta + np.log(r) - np.log(a)) * (t1 * eta / (eta + t1))
            else:
                v = (v / eta + np.log(c) - np.log(b)) * (t2 * eta / (eta + t2))

            # compute unreg objective
            B = get_B(u, v, C)
            rkl = get_kl(B, r, axis=1)
            ckl = get_kl(B, c, axis=0)
            reg_obj = dotp(C, B) + t1 * rkl + t2 * ckl - eta * get_entropy(B)
            reg_obj_list.append(reg_obj)
            unreg_obj = dotp(C, B) + t1 * rkl + t2 * ckl
            unreg_obj_list.append(unreg_obj)

            # compute gradient norm 1
            grad_norm = get_grad_norm(u, v, C, r, c, squared=False)
            grad_norm_list.append(grad_norm)

            # function value
            # need to recompute B!
            f_val = f(u, v, C, r, c)
            f_val_list.append(f_val)

            
            # new ratio
            f_val_diff = f_val_list[-2] - f_val_list[-1]

            kl1 = get_KL(r * np.exp(-u / t1), a)

            kl2 = get_KL(c * np.exp(-v / t2), b)
            log_ratio_list.append(f_val_diff / (kl1 + kl2) / eta)

            if np.abs(f_val_diff) < 1e-10: break


    eta_ratio_list.append(np.min(log_ratio_list))

plt.plot(eta_list, eta_ratio_list)
plt.show()
