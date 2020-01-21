import cvxpy as cp
import numpy as np 
import matplotlib.pyplot as plt 
from UOT import * 
nr = 20
nc = 20
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
C = C / sum(C) * 15
a = np.random.uniform(low=0.1, high=1, size=nr)

b = np.random.uniform(low=0.1, high=1, size=nc)
print(a.sum(), b.sum())

tau = 1
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
# print(X.value)


def get_eta_k(eps):
    alpha = sum(a)
    beta = sum(b)
    # S = (alpha + beta + 1 / np.log(nr)) * (2 * tau + 2)
    S = (alpha + beta) + 0.25 / np.log(nr)
    # T = 4 * ((alpha + beta) * (np.log(alpha + beta) + np.log(nr)) + 1)
    T = 2 * ((alpha + beta) * (0.5 * np.log(alpha + beta) - 0.5 * np.log(2) + np.log(nr) - 1) + 5/4)
    U = np.max([S + T, eps, 4 * eps * np.log(nr) / tau])
    eta = eps / U

    R = np.log(a).max() + np.log(b).max() + np.max([np.log(nr), supnorm(C) / eta - np.log(nr)])
    k = np.e * tau * U / eps * (np.log(6) + np.log(eta) + np.log(R) + np.log(tau) + np.log(tau+1) + np.log(U) - np.log(eps))

    print(U, k)

    return eta, k


eps_list = np.linspace(1, 0.1, 10)
eta_list = []
k_list = []

for eps in eps_list:
    eta, k = get_eta_k(eps)
    eta_list.append(eta)
    k_list.append(k)

eta_list = np.array(eta_list)
print(eta_list)

stop_iter_list = []
for eps, eta in zip(eps_list, eta_list):

    u, v, info = sinkhorn_uot(C, a.reshape(nr,1), b.reshape(nc,1), eta=eta, t1=tau, t2=tau, n_iter=10000, early_stop=False, eps=eps, opt_val=prob.value)
    stop_iter_list.append(info['stop_iter'])
    
# plt.plot([0, 1000], [prob.value, prob.value], c='red')
fig, ax = plt.subplots(1,1)
xs = np.arange(10)
ax.plot(xs, stop_iter_list, label=f'main curve, opt val={prob.value:.3f}')
eps_list_str = ['{:.2f}'.format(x) for x in eps_list]
ax.set_xticklabels(eps_list_str)

# from matplotlib.ticker import FormatStrFormatter
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.set_xlabel('epsilon')
ax.set_ylabel('k (iteration)')
inv_eps = 1 / eps_list
inv_eps = stop_iter_list[0] / inv_eps[0] * inv_eps

inv_eps_2 = 1 / eps_list**2
inv_eps_2 = stop_iter_list[0] / inv_eps_2[0] * inv_eps_2

# inv_eps_log = 1 / eps_list * np.log(1 / eps_list)
# inv_eps_log = stop_iter_list[0] / inv_eps_log[0] * inv_eps_log

ax.plot(xs, inv_eps, label='$1 / \epsilon$')
ax.plot(xs, inv_eps_2, label='$1 / \epsilon^{2}$')
# ax.plot(xs, inv_eps_log, label='$1 / \epsilon * \log (1 / \epsilon)$')


k_list = np.array(k_list)
# k_list = stop_iter_list[0] / k_list[0] * k_list
ax.plot(xs, k_list, label='formula')

ax.legend()
plt.show()

