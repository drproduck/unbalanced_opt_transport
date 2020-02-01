import cvxpy as cp
import numpy as np 
import numpy
import matplotlib.pyplot as plt 
from UOT import * 

np.random.seed(999)
def get_eta_k(eps):
    alpha = sum(a)
    beta = sum(b)
    # S = (alpha + beta + 1 / np.log(nr)) * (2 * tau + 2)
    S = 0.5 * (alpha + beta) + 0.5 + 0.25 / np.log(nr)
    # T = 4 * ((alpha + beta) * (np.log(alpha + beta) + np.log(nr)) + 1)
    T = 0.5 * (alpha + beta) * (np.log(alpha + beta) - np.log(2) + np.log(nr) - 0.5) + np.log(nr) + 5/2
    
    print('S, T:', S, T)

    U = np.max([S + T, 2 * eps, 4 * eps * np.log(nr) / tau, 4 * eps * (alpha+beta) * np.log(nr) / tau])
    eta = eps / U

    R = np.log(a).max() + np.log(b).max() + np.max([np.log(nr), supnorm(C) / eta - np.log(nr)])
    k = (tau * U / eps + 1) * (np.log(8) + np.log(eta) + np.log(R) + np.log(tau) + np.log(tau+1) + 3 * np.log(U) - 3 * np.log(eps))

    return eta, k

eta_list = []
k_list = []
n_list = [10, 20, 30, 40, 50, 60, 70]
stop_iter_list = []

for n in n_list:
    nr = n
    nc = n
    C = np.random.uniform(low=1, high=50, size=(nr, nc)).astype(numpy.longdouble)
    C = (C + C.T) / 2

    print('sum C:', C.sum())
    a = np.random.uniform(low=0.1, high=1.0, size=nr).astype(numpy.longdouble)
    a = a / a.sum() * 2
    b = np.random.uniform(low=0.1, high=1.0, size=nc).astype(numpy.longdouble)
    b = b / b.sum() * 4
    print('sum a:', a.sum(), 'sum b:', b.sum())

    tau = 5
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

    eps = 0.1
    eta, k = get_eta_k(eps)
    eta_list.append(eta)
    k_list.append(k)
    print('eps, eta, k:', eps, eta, k)


    u, v, info = sinkhorn_uot(C, a.reshape(nr,1), b.reshape(nc,1), eta=eta, t1=tau, t2=tau, n_iter=100000, early_stop=False, eps=eps, opt_val=prob.value)
    stop_iter_list.append(info['stop_iter'])
        

# plt.plot([0, 1000], [prob.value, prob.value], c='red')
fig, ax = plt.subplots(1,1)
# xs = np.arange(n_eps)
xs = n_list

ax.set_xlabel('n')
ax.set_ylabel('k (iterations)')

k_list = np.array(k_list)
# k_list = stop_iter_list[0] / k_list[0] * k_list
ax.plot(xs, k_list, label='Our bound ($k_f$)')
ax.plot(xs, stop_iter_list, label='max. no. iterations ($k_e$)')

# ax.set_xticklabels(list([f"{x:.3f}" for x in np.linspace(1, 0.1, 10)]))
# plt.xticks(list([f"{x:.3f}" for x in np.linspace(1, 0.1, 10)]))
# print(list([f"{x:.3f}" for x in np.linspace(1, 0.1, 10)]))


ax.legend()
plt.show()

