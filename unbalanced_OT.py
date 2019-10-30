import numpy as np
import matplotlib.pyplot as plt
import pdb
import warnings

np.seterr(all='warn')

# hyper parameters
t1 = 1
t2 = 1
eta = 0.1
min_log_ratio_list = []

rep_str = ''
rep_str += f'tau_1={t1:.3f}, tau_2={t2:.3f}, eta={eta:.3f}\n'

# def softmin(X, axis, eta):


def get_entropy(P):
    logP = np.log(P)
    return -1 * np.sum(logP * P - P)

def get_kl(B, a, axis):
    xB = B.sum(axis=axis).reshape(-1, 1)
    log_ratio = np.log(xB) - np.log(a)
    ret = xB * log_ratio - xB + a
    return ret.sum()

def get_B(u, v, C):
    n, m = C.shape
    K = -C + u @ np.ones((1, m)) + np.ones((n, 1)) @ v.T
    return np.exp(K / eta)

def f(u, v, C, r, c):
    B = get_B(u, v, C)
    f = eta * B.sum() + t1 * np.exp(- u / eta).T @ r + t2 * np.exp(- v / eta).T @ c
    f = f.squeeze()

    return f

def get_grad_norm(u, v, C, r, c):
    try:
        B = get_B(u, v, C)
        rB = B.sum(axis=1).reshape(-1, 1)
        ur = np.exp(- u / t1) * r
        Du = rB - ur
        Du = np.abs(Du).sum().squeeze()

        cB = B.sum(axis=0).reshape(-1, 1)
        vc = np.exp(- v / t2) * c
        Dv = cB - vc
        Dv = np.abs(Dv).sum().squeeze()

    except Warning:
        pdb.set_trace()

    return Du**2 + Dv**2

# for n_pts in np.arange(1, 100):
for n_pts in [100]:
    col_ones = np.ones((n_pts, 1))

    # problem-specific parameters (random)
    r = np.random.uniform(size=(n_pts, 1))
    rep_str += f'a={r}\n'
    c = np.random.uniform(size=(n_pts, 1))
    rep_str += f'b={c}\n'
    C = np.random.uniform(low=1, high=10, size=(n_pts, n_pts))
    C = (C + C.T) / 2
    rep_str += f'C={C}\n'

    # initial solution
    u = np.zeros((n_pts, 1))
    v = np.zeros((n_pts, 1))

    # update values
    grad_norm_list = []
    f_val_list = []
    log_ratio_list = []
    unreg_obj_list = []
    reg_obj_list = []
    n_iter = 100


    # compute before any updates
    f_val = f(u, v, C, r, c)
    grad_norm = get_grad_norm(u, v, C, r, c)
    f_val_list.append(f_val)
    grad_norm_list.append(grad_norm)

    for i in range(n_iter):

        B = get_B(u, v, C)
        a = (B @ col_ones).reshape(-1, 1)
        b = (B.T @ col_ones).reshape(-1, 1)

        # update
        if i % 2 == 0:
            u = (u / eta + np.log(r) - np.log(a)) * (t1 * eta / (eta + t1))
        else:
            v = (v / eta + np.log(c) - np.log(b)) * (t2 * eta / (eta + t2))

        # compute unreg objective
        B = get_B(u, v, C)
        rkl = get_kl(B, r, axis=1)
        ckl = get_kl(B, c, axis=0)
        reg_obj = np.sum(C * B) + t1 * rkl + t2 * ckl - eta * get_entropy(B)
        reg_obj_list.append(reg_obj)
        unreg_obj = np.sum(C * B) + t1 * rkl + t2 * ckl
        unreg_obj_list.append(unreg_obj)

        # compute gradient
        grad_norm = get_grad_norm(u, v, C, r, c)
        grad_norm_list.append(grad_norm)

        # function value
        # need to recompute B!
        f_val = f(u, v, C, r, c)
        f_val_list.append(f_val)

        # unreg_obj = f_val - eta * get_entropy(B)
        # unreg_obj_list.append(unreg_obj)

        f_val_diff = f_val_list[-2] - f_val_list[-1]
        # log_ratio = np.log(f_val_diff) - np.log(grad_norm_list[-2])
        log_ratio = f_val_diff / grad_norm_list[-2]
        log_ratio_list.append(log_ratio)

    min_log_ratio_list.append(log_ratio)

# rep_str += f'min log ratio={log_ratio}\n'
rep_str = ''

ig, ax = plt.subplots(1,5)
ax[0].plot(np.arange(n_iter), log_ratio_list)
ax[0].set_title('$(f^k - f^{k+1}) / (||du f(u^k, v^k)||_1^2 + ||dv f(u^k, v^k)||_1^2)$')
ax[1].plot(np.arange(n_iter+1), grad_norm_list)
ax[1].set_title('$(||du f(u^k, v^k)||_1^2 + ||dv f(u^k, v^k)||_1^2)$')
ax[2].plot(np.arange(n_iter+1), f_val_list, label=rep_str)
ax[2].legend()
ax[2].set_title('$f^k$')
ax[3].plot(np.arange(n_iter), reg_obj_list)
ax[3].set_title('regularized primal')
ax[4].plot(np.arange(n_iter), unreg_obj_list)
ax[4].plot('unregularized')

# print(min_log_ratio_list)
# plt.plot(np.arange(1,100), min_log_ratio_list)
plt.show()
