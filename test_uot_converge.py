import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb

def get_stats(li):
    return li[-1], np.min(li)
nr = 10
nc = 10
C = np.random.uniform(low=0.1, high=1, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))


fig, ax = plt.subplots(3,1)

eta_list = np.linspace(0.01, 5, 20)

converge_dual = []
converge_primal = []
gap_unreg = []

for i, eta in enumerate(eta_list):
    u, v, info = sinkhorn_uot(C, r, c, eta=eta, t1=10, t2=10, n_iter=2000)

    converge, min_val = get_stats(info['f_val_list'])
    converge_dual.append(converge)

    converge, min_val = get_stats(info['f_primal_val_list'])
    converge_primal.append(converge)

    converge, min_val = get_stats(info['unreg_f_val_list'])
    gap_unreg.append(converge - min_val)

    if i == 0:
        ax[2].scatter(eta, converge, c='red', s=5, label='converge val')
        ax[2].scatter(eta, min_val, c='green', s=5, label='min val')
    else:
        ax[2].scatter(eta, converge, c='red', s=10)
        ax[2].scatter(eta, min_val, c='green', s=10)

ax[0].set_title('dual')
ax[0].scatter(eta_list, converge_dual, label='converge val', s=10, c='red')
# ax[0].plot(eta_list, converge_dual)
ax[1].set_title('primal')
ax[1].scatter(eta_list, converge_primal, label='converge val', s=10, c='red')
# ax[1].plot(eta_list, converge_primal)
ax[2].set_title('unregularized')
ax[2].plot(eta_list, gap_unreg, label='converge val - min val')
ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()
