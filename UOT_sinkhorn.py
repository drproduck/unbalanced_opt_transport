import numpy as np
import matplotlib.pyplot as plt
import pdb
import warnings
from copy import copy


# hyper parameters
t1 = 1.0
t2 = 1.0
eta = 1.0

rep_str = ''
# rep_str += f'tau_1={t1:.3f}, tau_2={t2:.3f}, eta={eta:.3f}\n'



def get_entropy(P):
    logP = np.log(P + 1e-20)
    return -1 * np.sum(logP * P - P)


def get_KL(P, Q):
    log_ratio = np.log(P + 1e-20) - np.log(Q + 1e-20)
    return np.sum(P * log_ratio - P + Q)

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

def norm1(x, y):
    return np.sum(np.abs(x - y))

def get_grad_norm(u, v, C, r, c, square=False):
    B = get_B(u, v, C)
    rB = B.sum(axis=1).reshape(-1, 1)
    ur = np.exp(- u / t1) * r
    Du = rB - ur
    Du = np.abs(Du).sum()

    cB = B.sum(axis=0).reshape(-1, 1)
    vc = np.exp(- v / t2) * c
    Dv = cB - vc
    Dv = np.abs(Dv).sum()

    if square:
        return Du**2 + Dv**2
    else:
        return Du + Dv

def log_sinkhorn(
    for n_pts in [10]:
        col_ones = np.ones((n_pts, 1))

        # problem-specific parameters (random)
        r = np.random.uniform(size=(n_pts, 1))
        # r = np.array([[1],[3]])
        rep_str += f'a={r}\n'
        c = np.random.uniform(size=(n_pts, 1))
        # c = np.array([[2],[4]])

        r = r / r.sum() * 5
        c = c / c.sum() * 5
        rep_str += f'b={c}\n'
        C = np.random.uniform(low=1, high=10, size=(n_pts, n_pts))
        C = (C + C.T) / 2
        # C = np.array([[8, 6],[6, 8]])
        rep_str += f'C={C}\n'

        # initial solution
        u = np.zeros((n_pts, 1))
        v = np.zeros((n_pts, 1))

        # update values
        grad_norm_list = []
        f_val_list = []
        norm_ratio_list = []
        kl_ratio_list = []
        unreg_obj_list = []
        reg_obj_list = []
        n_iter = 1000


        # compute before any updates
        f_val = f(u, v, C, r, c)
        grad_norm = get_grad_norm(u, v, C, r, c)
        f_val_list.append(f_val)
        grad_norm_list.append(grad_norm)

        for i in range(n_iter):

            B = get_B(u, v, C)
            a = B.sum(axis=1).reshape(-1, 1)
            b = B.sum(axis=0).reshape(-1, 1)

            u_old = copy(u)
            v_old = copy(v)
            # update
            if i % 2 == 0:
                u = (u / eta + np.log(r) - np.log(a)) * (t1 * eta / (eta + t1))
            else:
                v = (v / eta + np.log(c) - np.log(b)) * (t2 * eta / (eta + t2))

            # compute unreg objective
            # B = get_B(u, v, C)
            # rkl = get_kl(B, r, axis=1)
            # ckl = get_kl(B, c, axis=0)
            # reg_obj = dotp(C, B) + t1 * rkl + t2 * ckl - eta * get_entropy(B)
            # reg_obj_list.append(reg_obj)
            # unreg_obj = dotp(C, B) + t1 * rkl + t2 * ckl
            # unreg_obj_list.append(unreg_obj)

            # compute gradient
            grad_norm = get_grad_norm(u, v, C, r, c, square=True)
            grad_norm_list.append(grad_norm)

            # function value
            # need to recompute B!
            f_val = f(u, v, C, r, c)
            f_val_list.append(f_val)

            # unreg_obj = f_val - eta * get_entropy(B)
            # unreg_obj_list.append(unreg_obj)

            f_val_diff = f_val_list[-2] - f_val_list[-1]
            norm_ratio = f_val_diff / grad_norm_list[-2] / eta
            norm_ratio_list.append(norm_ratio)

            # KL ratio
            kl1 = get_KL(r * np.exp(- u_old / t1), a)
            kl2 = get_KL(c * np.exp(- v_old / t2), b)
            kl_ratio = f_val_diff / (kl1 + kl2) / eta
            kl_ratio_list.append(kl_ratio)

            # early stop
            if f_val_diff < 1e-10:
                break

        min_norm_ratio_list.append(np.min(norm_ratio_list))
        min_kl_ratio_list.append(np.min(kl_ratio_list))

    # eta_log_ratio_list.append(copy(min_log_ratio_list))
# rep_str += f'min log ratio={log_ratio}\n'
rep_str = ''

plt.plot(eta_list, min_norm_ratio_list, label='norm_1^2 ratio')
plt.plot(eta_list, min_kl_ratio_list, label='kl ratio')
plt.xlabel('eta')
plt.legend()
plt.show()

# fig, ax = plt.subplots(1,5)
# ax[0].plot(np.arange(n_iter), log_ratio_list)
# ax[0].set_title('$(f^k - f^{k+1}) / (||du f(u^k, v^k)||_1^2 + ||dv f(u^k, v^k)||_1^2)$')
# ax[1].plot(np.arange(n_iter+1), grad_norm_list)
# ax[1].set_title('grad norm squared')
# ax[2].plot(np.arange(n_iter+1), f_val_list, label=rep_str)
# ax[2].set_title('$f^k$ (regularized dual)')
# 
# start, end = ax[2].get_ylim()
# ax[2].set_yticks(np.linspace(start, end, 20))
# 
# ax[3].plot(np.arange(n_iter), reg_obj_list)
# ax[3].set_title('regularized primal')
# 
# start, end = ax[3].get_ylim()
# ax[3].set_yticks(np.linspace(start, end, 20))
# 
# ax[4].plot(np.arange(n_iter), unreg_obj_list)
# ax[4].set_title('unregularized')
# print(f_val_list)
# print(reg_obj_list)

# print(min_log_ratio_list)
# plt.plot(np.arange(1,100), min_log_ratio_list)
# plt.plot(np.arange(1,100), -1 * np.log(np.arange(1,100)))
# plt.title(f'eta={eta:.3f}, tau1={t1:.3f}, tau2={t2:.3f}')
# plt.show()


# print(len(eta_log_ratio_list[0]))
# fix, ax = plt.subplots(2,5)
# for i in range(10):
#     ax[i//5, i%5].plot(np.arange(1,100), eta_log_ratio_list[i])
#     ax[i//5, i%5].plot(np.arange(1,100), -1 * np.log(1 / np.arange(1, 100))**2)
#     ax[i//5, i%5].set_title(eta_list[i]) 
# plt.show()
