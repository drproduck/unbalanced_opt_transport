import numpy as np
import matplotlib.pyplot as plt
from OT import *
import pdb

nr = 10
nc = 10
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2

r = np.random.uniform(low=0.1, high=1, size=(nr, 1))
r = r / r.sum()
c = np.random.uniform(low=0.1, high=1, size=(nc, 1))
c = c / c.sum()
print(sum(r), sum(c))

u, v, info = sinkhorn_ot(C, r, c, eta=0.01, n_iter=10000)

fig, ax = plt.subplots(3,1)

converge = info['f_val_list'][-1]
max_val = np.max(info['f_val_list'])
ax[0].plot(np.arange(info['stop_iter']+1), info['f_val_list'], label=f'dual, converge={converge:.3f}, max={max_val:.3f}')

converge = info['f_primal_val_list'][-1]
max_val = np.max(info['f_primal_val_list'])
ax[1].plot(np.arange(info['stop_iter']+1), info['f_primal_val_list'], label=f'primal, converge={converge:.3f}, max={max_val:.3f}')

converge = info['unreg_f_val_list'][-1]
max_val = np.max(info['unreg_f_val_list'])
ax[2].plot(np.arange(info['stop_iter']+1), info['unreg_f_val_list'], label=f'unregularized, converge={converge:.3f}, max={max_val:.3f}')
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.legend()
plt.show()
