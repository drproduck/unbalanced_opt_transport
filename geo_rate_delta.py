import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb
import cvxpy as cp

np.random.seed(9999)

nr = 100
nc = 100
eta = 1
tau = 5
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

def solve_g_dual_cp(C, a, b, eta, tau):
    u = cp.Variable(shape=a.shape)
    v = cp.Variable(shape=b.shape)

    u_stack = cp.vstack([u.T for _ in range(nr)])
    v_stack = cp.hstack([v for _ in range(nc)])
    print(u_stack.shape, v_stack.shape)

    # obj = eta * cp.sum(cp.multiply(cp.exp(u + v.T) * cp.exp(v).T, 1 / cp.exp(C)))
    # obj = eta * cp.sum(cp.multiply(cp.exp(u_stack + v_stack), 1 / cp.exp(C)))
    obj = eta * cp.sum(cp.exp((u_stack + v_stack - C) / eta))
    obj += tau * cp.sum(cp.multiply(cp.exp(- u / tau), a))
    obj += tau * cp.sum(cp.multiply(cp.exp(- v / tau), b))

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()

    return prob.value, u.value, v.value

opt_val, ustar, vstar = solve_g_dual_cp(C, r, c, eta, tau)
print(ustar.shape, vstar.shape)
ustar = ustar.reshape(-1, 1)
vstar = vstar.reshape(-1, 1)

u, v, info = sinkhorn_uot(C, r, c, eta=eta, t1=tau, t2=tau, early_stop=False, n_iter=1000)

delta_list = []
geom_list = []

max_ratio_list = []
ratio_list = []
bound_list = []


print(info['stop_iter'])
for i in range(info['stop_iter']):
    delta = np.max([supnorm(info['u_list'][i] - ustar), supnorm(info['v_list'][i] - vstar)])
    delta_list.append(delta)

    if i == 0:
        geom_list.append(delta)
        bound_list.append(tau * (np.max([supnorm(np.log(r)), supnorm(np.log(c))]) + np.max([supnorm(C) / eta - np.log(nr), np.log(nr)])))
    else:
        geom_list.append(geom_list[-1] * (tau / (tau + eta)))
        bound_list.append(bound_list[-1] * (tau / (tau + eta)))


    max_ratio = geom_list[-1] / delta_list[-1]
    max_ratio_list.append(max_ratio)

    if i > 0:
        ratio = delta_list[-2] / delta_list[-1]
        ratio_list.append(ratio)

# print(delta_list)
print(geom_list)
fig, ax = plt.subplots(2, 1)
plt.rcParams.update({'font.size': 22})
# plt.figure(figsize=(10, 8))
ax[0].plot(range(info['stop_iter']-1), ratio_list, linewidth=2, label='$\Delta_{k-1} / \Delta_{k}$')
ax[1].plot(range(info['stop_iter']+1), info['f_val_list'], linewidth=2)
# plt.plot(epsilon_list, [tmp / 1000 for tmp in k_list_empirical_true], "r", linewidth=4, label=r"$k_{true}$")
# plt.plot(epsilon_list, [tmp / 1000 for tmp in k_list_empirical_first], "g", linewidth=4, label=r"$k_{first}$")
# plt.plot(epsilon_list, [tmp / 1000 for tmp in k_list_formula], "b", linewidth=4, label=r"$k_{formula}$")
# plt.xlabel("epsilon")
plt.xlabel("k (iterations)")
plt.legend(prop={'size': 30})

plt.savefig('delta_rate.eps', bbox_inches='tight')
# plt.savefig('k_comparison.png', bbox_inches='tight')
plt.show()

