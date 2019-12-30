import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb
from utils import *

nr = 100
nc = 100


eta = 0.1
tau = 10
C = np.random.uniform(low=100, high=1000, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

eta_list = np.linspace(0.05, 1, 20)
delta_list = []
bound_list = []
bound2_list = []
diff_list = []
for eta in eta_list:

    u, v, info = sinkhorn_uot(C, r, c, eta=eta, t1=tau, t2=tau, n_iter=2000, early_stop=True)
    delta = np.max([supnorm(u), supnorm(v)])
    delta_list.append(delta)
    # bound = tau * (supnorm(np.log(r)) + supnorm(np.log(c)) + np.max([np.log(100), supnorm(C) / eta - np.log(100)]))
    bound = tau * (supnorm(np.log(r)) + supnorm(np.log(c)) + np.max([np.log(100), supnorm(C)]))
    # bound = tau * (supnorm(np.log(r)) + supnorm(np.log(c)) + np.log(100))
    # bound = tau * (supnorm(np.log(r)) + supnorm(np.log(c)))
    bound2 = 2 * eta * np.max([supnorm(np.log(r)), supnorm(np.log(c))]) + eta * np.log(100) + np.min(C) + np.max(C)
    if supnorm(C) / eta - np.log(100) > np.log(100):
        print('use C / eta')

    else: print('use log(n)')

    bound_list.append(bound)
    bound2_list.append(bound2)
    diff = bound - delta
    diff_list.append(diff)

fig, ax = plt.subplots(1,1)

ax.plot(eta_list, delta_list, label='delta')
ax.plot(eta_list, bound_list, label='bound')
ax.plot(eta_list, bound2_list, label='bound2')
# ax.plot(eta_list, diff_list)
ax.legend()
# print(supnorm(C))
print(delta_list)
# print(bound_list)
print(diff_list)

plt.show()
