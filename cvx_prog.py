import cvxpy as cp
import numpy as np 
import matplotlib.pyplot as plt 
from UOT import * 
nr = 100
nc = 100
C = np.random.uniform(low=10, high=100, size=(nr, nc))
C = (C + C.T) / 2
a = np.random.uniform(low=1, high=10, size=nr)

b = np.random.uniform(low=1, high=10, size=nc)

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
print(X.value)


def get_eta(eps):
    alpha = sum(a)
    beta = sum(b)
    S = (alpha + beta + 1 / np.log(nr)) * (2 * tau + 2)
    T = 4 * ((alpha + beta) * (np.log(alpha + beta) + np.log(nr)) + 1)
    U = np.max([S + T, eps, 2 * eps * np.log(nr) / tau])
    eta = eps / U

    return eta


eps_list = np.linspace(10, 1, 100)
eta_list = []
for eps in eps_list:
    eta_list.append(get_eta(eps))
eta_list = np.array(eta_list)

stop_iter_list = []
for eps, eta in zip(eps_list, eta_list):

    u, v, info = sinkhorn_uot(C, a.reshape(nr,1), b.reshape(nc,1), eta=0.1, t1=tau, t2=tau, n_iter=10000, early_stop=False, eps=eps, opt_val=prob.value)
    stop_iter_list.append(info['stop_iter'])
    
# plt.plot([0, 1000], [prob.value, prob.value], c='red')
fig, ax = plt.subplots(1,1)
xs = np.arange(100)
ax.plot(xs, stop_iter_list, label=f'main curve, opt val={prob.value:.3f}')
eps_list_str = ['{:.2f}'.format(x) for x in eps_list]
ax.set_xticklabels(eps_list_str)

# from matplotlib.ticker import FormatStrFormatter
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.set_xlabel('epsilon')
ax.set_ylabel('k (iteration)')
inv_eps = 1 / eps_list
inv_eps = stop_iter_list[0] / inv_eps[0] * inv_eps

inv_eps_2 = 1 / eps_list**0.5
inv_eps_2 = stop_iter_list[0] / inv_eps_2[0] * inv_eps_2

ax.plot(xs, inv_eps, label='$1 / \epsilon$')
ax.plot(xs, inv_eps_2, label='$1 / \epsilon^{0.5}$')

ax.legend()
plt.show()

