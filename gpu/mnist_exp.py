# import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt 
# from UOT import *
from UOT_torch import *
from copy import copy
from download_mnist import _load_mnist
from prog import exact_uot
from time import time

import sys

USE_PRESET_PAIRS = True
# USE_PRECOMPUTED_OPTVAL = True

def repeat_newdim(x, n_repeat, newdim):
     x = np.expand_dims(x, axis=newdim)
     x = np.repeat(x, n_repeat, newdim)
     return x


def get_L1_C(dim):
    xv, yv = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
    vv1 = np.concatenate((xv.reshape(-1,1), yv.reshape(-1, 1)), axis=1)

    vv2 = copy(vv1)
    vv1 = repeat_newdim(vv1, dim**2, 1)
    vv2 = repeat_newdim(vv2, dim**2, 0)
    # print(vv1.shape, vv2.shape)
    C = vv1 - vv2
    C = np.abs(C).sum(axis=-1)
    return C

def get_eta_k(eps, a, b, tau, nr):
    alpha = np.sum(a)
    beta = np.sum(b)
    # S = (alpha + beta + 1 / np.log(nr)) * (2 * tau + 2)
    S = 0.5 * (alpha + beta) + 0.5 + 0.25 / np.log(nr)
    # T = 4 * ((alpha + beta) * (np.log(alpha + beta) + np.log(nr)) + 1)
    T = 0.5 * (alpha + beta) * (np.log(alpha + beta) - np.log(2) + np.log(nr) - 0.5) + np.log(nr) + 5/2

    U = np.max([S + T, 2 * eps, 4 * eps * np.log(nr) / tau, 4 * eps * (alpha+beta) * np.log(nr) / tau])
    eta = eps / U

    # print('S, T, U, eps, eta:', S, T, U, eps, eta)

    def supnorm(X):
        return np.max(np.abs(X))

    R = np.log(a).max() + np.log(b).max() + np.max([np.log(nr), supnorm(C) / eta - np.log(nr)])
    k = (tau * U / eps + 1) * (np.log(8) + np.log(eta) + np.log(R) + np.log(tau) + np.log(tau+1) + 3 * np.log(U) - 3 * np.log(eps))

    return eta, k
    
C = get_L1_C(28).astype(np.float64)
# print(C)

x, y = _load_mnist('.', split_type='train', download=True)
# pairs = ((3, 5), (20562, 12428), (2564, 12380), (48485, 7605), (26428, 42698), (6152, 25061), (13168, 7506), (40816, 39370), (846, 16727), (31169, 7144))
pairs = ((3, 5), )

# pre-computed optimal values. Use this if you don't want to run again, or if you use other image pairs.
# opt_val_list = (
# 321.83947656441, 
# 167.64555073518932,
# 207.3502556064093,
# 423.35617236431176,
# 117.52384257739209,
# 459.297428039594,
# 138.082535273433,
# 256.3855206718765,
# 221.13453306061183,
# 348.8398254732447
# )


k_list_empirical_first = []
k_list_formula = []

eps_list = np.linspace(5.0, 0.5, 10)
tau = 10

for (id1, id2) in pairs:
    print('pair', id1, id2)
    a = x[id1].astype(np.float64)
    b = x[id2].astype(np.float64)
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    a[a == 0] = 1e-6
    b[b == 0] = 1e-6

    # start = time()
    # uot_opt_val = exact_uot(C, a.flatten(), b.flatten(), tau, vbo=True)
    # print('time elapsed:', time() - start)
    # print('exact val:', uot_opt_val)

    for eps in eps_list:
        start = time()
        eta, k = get_eta_k(eps, a, b, tau, 784)
        k_list_empirical_first.append(k)
        print('eps:', eps, 'eta:', eta, 'k:', k)
        # u, v, info = sinkhorn_uot(C, a, b, eta=eta, t1=tau, t2=tau, n_iter=10000000, eps=eps, opt_val=uot_opt_val, vbo=True)
        # print('time elapsed:', time() - start)
        # print('approx val:', info['unreg_f_val_list'][-1])
        # print(info['stop_iter'])
        # k_list_formula.append(info['stop_iter'])

sys.exit()
k_list = np.array(k_list)
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(10, 8))
plt.plot(eps_list, [tmp / 1000 for tmp in k_list_empirical_first], "g", linewidth=4, label=r"$k_{first}$")
plt.plot(eps_list, [tmp / 1000 for tmp in k_list_formula], "b", linewidth=4, label=r"$k_{formula}$")
plt.xlabel("epsilon")
plt.ylabel("k (thousand iterations)")
plt.legend(prop={'size': 30})

plt.savefig('k_comparison.eps', bbox_inches='tight')
plt.savefig('k_comparison.png', bbox_inches='tight')
plt.show()

