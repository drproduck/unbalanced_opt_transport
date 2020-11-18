import numpy as np
import matplotlib.pyplot as plt
from OT import *
import pdb

nr = 10
nc = 10
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
n_iter = 100

r = np.random.uniform(low=0.1, high=1, size=(nr, 1))
r = r / r.sum()
c = np.random.uniform(low=0.1, high=1, size=(nc, 1))
c = c / c.sum()
print(sum(r), sum(c))

u, v, S, info = sinkhorn(C, r, c, eta=0.01, n_iter=n_iter, debug=True, early_stop=None)
print(S.sum(1).flatten())
print(S.sum(0).flatten())

fig, ax = plt.subplots(2,1)

converge = info['f_reg_dual'][-1]
max_val = np.max(info['f_reg_dual'])
ax[0].plot(np.arange(n_iter+1), info['f_reg_dual'], label=f'dual, converge={converge:.3f}, max={max_val:.3f}')

converge = info['f_reg_primal'][-1]
min_val = np.min(info['f_reg_primal'])
ax[0].plot(np.arange(n_iter+1), info['f_reg_primal'], label=f'primal, converge={converge:.3f}, max={min_val:.3f}')

converge = info['f_unreg'][-1]
max_val = np.max(info['f_unreg'])
ax[1].plot(np.arange(n_iter+1), info['f_unreg'], label=f'unregularized, converge={converge:.3f}, max={max_val:.3f}')
ax[0].legend()
ax[1].legend()
plt.legend()
plt.show()
