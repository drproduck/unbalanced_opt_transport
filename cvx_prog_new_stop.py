import cvxpy as cp
import numpy as np 
import numpy
import matplotlib.pyplot as plt 
from UOT import * 
from UOT_newstop import *

np.random.seed(9999)

nr = 50
nc = 50
n_eps = 10
C = np.random.uniform(low=1, high=50, size=(nr, nc)).astype(numpy.longdouble)
print(C.max())
C = (C + C.T) / 2
# C = C / sum(C) * 15
print('sum C:', C.sum())
a = np.random.uniform(low=0.1, high=1, size=nr).astype(numpy.longdouble)
a = a / sum(a) * 2
b = np.random.uniform(low=0.1, high=1, size=nc).astype(numpy.longdouble)
b = b / sum(b) * 4
print('sum a:', a.sum(), 'sum b:', b.sum())

tau = 10
X = cp.Variable((nr,nc), nonneg=True)


row_sums = cp.sum(X, axis=1)
col_sums = cp.sum(X, axis=0)

obj = cp.sum(cp.multiply(X, C))

obj -= tau * cp.sum(cp.entr(row_sums))
obj -= tau * cp.sum(cp.entr(col_sums))

obj -= tau * cp.sum(cp.multiply(row_sums, cp.log(a)))
obj -= tau * cp.sum(cp.multiply(col_sums, cp.log(b)))

obj -= 2 * tau * cp.sum(X)
obj += tau * cp.sum(a) + tau * cp.sum(b)

prob = cp.Problem(cp.Minimize(obj))

prob.solve()

print('optimal value', prob.value)


def get_eta_k(eps_list):
    alpha = sum(a)
    beta = sum(b)
    # S = (alpha + beta + 1 / np.log(nr)) * (2 * tau + 2)
    S = 0.5 * (alpha + beta) + 0.5 + 0.25 / np.log(nr)
    # T = 4 * ((alpha + beta) * (np.log(alpha + beta) + np.log(nr)) + 1)
    T = 0.5 * (alpha + beta) * (np.log(alpha + beta) - np.log(2) + np.log(nr) - 0.5) + np.log(nr) + 5/2
    
    print('S, T:', S, T)

    eta_list = []
    k_list = []
    for eps in eps_list:
        U = np.max([S + T, eps, 4 * eps * np.log(nr) / tau, 2 * eps * (alpha+beta) * np.log(nr) / tau])
        eta = eps / U

        R = np.log(a).max() + np.log(b).max() + np.max([np.log(nr), supnorm(C) / eta - np.log(nr)])
        k = (tau * U / eps + 1) * (np.log(6) + np.log(eta) + np.log(R) + np.log(tau) + np.log(tau+1) + 3 * np.log(U) - 3 * np.log(eps))

        eta_list.append(eta)
        k_list.append(k)

    eta_list = np.array(eta_list)
    k_list = np.array(k_list)
    return eta_list, k_list


eps_list = np.linspace(1, 0.1, n_eps)
eta_list, k_list = get_eta_k(eps_list)

stop_iter_list_1 = []
stop_iter_list_2 = []
f_newstop_list = []
for eps, eta in zip(eps_list, eta_list):

    u, v, info_1 = sinkhorn_uot(C, a.reshape(nr,1), b.reshape(nc,1), eta=eta, t1=tau, t2=tau, n_iter=20000, early_stop=False, eps=eps, opt_val=prob.value)
    u, v, info_2 = sinkhorn_uot_newstop(C, a.reshape(nr,1), b.reshape(nc,1), eta=eta, t1=tau, t2=tau, n_iter=20000, early_stop=False, eps=eps, opt_val=prob.value)
    stop_iter_list_1.append(info_1['stop_iter'])
    stop_iter_list_2.append(info_2['stop_iter'])
    f_newstop_list.append(info_2['unreg_f_val_list'][-1])
    
# print(info_2['unreg_f_val_list'])
# print(stop_iter_list_2)
# plt.plot([0, 1000], [prob.value, prob.value], c='red')
fig, ax = plt.subplots(3,1)
xs = np.arange(n_eps)
ax[0].plot(xs, stop_iter_list_1, label=f'empirical (main), opt val={prob.value:.3f}')
ax[0].plot(xs, stop_iter_list_2, label=f'new stopping rule')
eps_list_str = ['{:.2f}'.format(x) for x in eps_list]
ax[0].set_xticklabels(eps_list_str)

ax[0].set_xlabel('epsilon')
ax[0].set_ylabel('k (iteration)')
inv_eps = 1 / eps_list
# inv_eps = stop_iter_list[0] / inv_eps[0] * inv_eps

inv_eps_2 = 1 / eps_list**2
# inv_eps_2 = stop_iter_list[0] / inv_eps_2[0] * inv_eps_2

# inv_eps_log = 1 / eps_list * np.log(1 / eps_list)
# inv_eps_log = stop_iter_list[0] / inv_eps_log[0] * inv_eps_log

ax[0].plot(xs, inv_eps, label='$1 / \epsilon$')
ax[0].plot(xs, inv_eps_2, label='$1 / \epsilon^{2}$')
# ax.plot(xs, inv_eps_log, label='$1 / \epsilon * \log (1 / \epsilon)$')


k_list = np.array(k_list)
# k_list = stop_iter_list[0] / k_list[0] * k_list
ax[0].plot(xs, k_list, label='formula')

print(np.array(f_newstop_list) - prob.value)
ax[0].legend()

ax[1].plot(eps_list, np.array(f_newstop_list) - prob.value, label='$f^k - f^*$')
ax[1].legend()

ax[2].plot(eps_list, stop_iter_list_1, label=f'empirical (main), opt val={prob.value:.3f}')
ax[2].plot(eps_list, stop_iter_list_2, label=f'new stopping rule')
ax[2].legend()
plt.show()

