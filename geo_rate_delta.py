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

u, v, info = sinkhorn_uot(C, r, c, eta=eta, t1=tau, t2=tau, early_stop=False, n_iter=2000)

delta_list = []
geom_list = []

max_ratio_list = []
ratio_list = []
bound_list = []

print(info['stop_iter'])
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

print(delta_list)
# print(geom_list)

# print(info['stop_iter'])
fig, ax = plt.subplots(1,1)

ax.plot(range(info['stop_iter']-1), ratio_list[1::2], label='$\Delta_{k-1} / \Delta_{k}$')
ax.set_xlabel('k (iterations)')
ax.legend()

plt.show()
