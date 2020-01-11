import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from UOT import *

nr = 100
nc = 100
C = np.random.uniform(low=10, high=100, size=(nr, nc))
C = (C + C.T) / 2
a = np.random.uniform(low=0.1, high=1, size=nr)

b = np.random.uniform(low=0.1, high=1, size=nc)

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




# eps = 100
# alpha = sum(a)
# beta = sum(b)
# S = 9 * (alpha + beta) * (2 * tau + 1) + 1
# T = 4 * ((alpha + beta) * (np.log(alpha + beta) + np.log(10)) + 1)
# U = S + T + eps + 2 * eps * np.log(10) / tau
# eta = eps / U
# print(eta)


eps_list = np.linspace(0.9, 0.01, 100)
eta_list = eps_list / np.log(10)
stop_iter_list = []
for eps, eta in zip(eps_list, eta_list):

    u, v, info = sinkhorn_uot(C, a.reshape(nr,1), b.reshape(nc,1), eta=0.1, t1=tau, t2=tau, n_iter=10000, early_stop=False, eps=eps, opt_val=prob.value)
    stop_iter_list.append(info['stop_iter'])
    
# plt.plot([0, 1000], [prob.value, prob.value], c='red')
print(eps_list)
print(stop_iter_list)
fig, ax = plt.subplots(1,1)
xs = np.arange(100)
ax.plot(xs, stop_iter_list)
start = eps_list[0]
start = 1 / start * np.log(1 / start)
a = stop_iter_list[0] / start
ax.plot(xs, a / eps_list * np.log(1 / eps))
ax.set_xticklabels(eps_list)
plt.show()

