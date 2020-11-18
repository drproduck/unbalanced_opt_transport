import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb

np.random.seed(9999)

nr = 100
nc = 100
eta = 0.1
tau = 10
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

u, v, info = sinkhorn_uot(C, r, c, eta=eta, t1=tau, t2=tau, n_iter=1000)

delta_list = []
geom_list = []

max_ratio_list = []
ratio_list = []
bound_list = []


for i in range(info['stop_iter']):
    delta = np.max([supnorm(info['u_list'][i] - info['u_list'][-1]), supnorm(info['v_list'][i] - info['v_list'][-1])])
    delta_list.append(delta)

    if i == 0:
        geom_list.append(delta)
        bound_list.append(tau * (np.max([supnorm(np.log(r)), supnorm(np.log(c))]) + np.max([supnorm(C) / eta - np.log(nr), np.log(nr)])))
        print(bound_list[-1] / geom_list[-1])
    else:
        geom_list.append(geom_list[-1] * (tau / (tau + eta)))
        bound_list.append(bound_list[-1] * (tau / (tau + eta)))


    max_ratio = geom_list[-1] / delta_list[-1]
    max_ratio_list.append(max_ratio)

    if i > 0:
        ratio = delta_list[-2] / delta_list[-1]
        ratio_list.append(ratio)

# print(delta_list)
# print(geom_list)

# print(info['stop_iter'])
fig, ax = plt.subplots(2,1)
ax[0].plot(list(range(info['stop_iter'])), delta_list, label='$\Delta^k = \max \{ ||u^k - u^*||_{\infty}, ||v^k - v^*||_{\infty} \}$')
ax[0].plot(list(range(info['stop_iter'])), geom_list, label='$\Delta_0 (\\frac{\\tau}{\\tau + \eta})^k$')
ax[0].legend()
ax[0].set_xlabel('k (iteration)')

# ax[1].plot(range(info['stop_iter']), max_ratio_list, label='orange / blue')
ax[1].plot(range(info['stop_iter']-1), ratio_list, label='$\Delta_{k-1} / \Delta_{k}$')
ax[1].legend()

ax[0].plot(range(info['stop_iter']), bound_list, label='$\\tau \\times R (\\frac{\\tau}{\\tau + \eta})^k$')
ax[0].legend()
plt.show()