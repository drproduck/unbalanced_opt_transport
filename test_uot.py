import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb
from time import time

nr = 1000
nc = 1000
C = np.random.uniform(low=10, high=100, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

start = time()
u, v, info = sinkhorn_uot(C, r, c, eta=0.1, t1=1, t2=1, n_iter=10000)
print('time elapsed:', time() - start)

fig, ax = plt.subplots(3,1)

converge = info['f_val_list'][-1]
min_val = np.min(info['f_val_list'])
ax[0].plot(np.arange(info['stop_iter']+1), info['f_val_list'], label=f'dual, converge={converge:.3f}, min={min_val:.3f}')

converge = info['f_primal_val_list'][-1]
min_val = np.min(info['f_primal_val_list'])
ax[1].plot(np.arange(info['stop_iter']+1), info['f_primal_val_list'], label=f'primal, converge={converge:.3f}, min={min_val:.3f}')

converge = info['unreg_f_val_list'][-1]
min_val = np.min(info['unreg_f_val_list'])
ax[2].plot(np.arange(info['stop_iter']+1), info['unreg_f_val_list'], label=f'unregularized, converge={converge:.3f}, min={min_val:.3f}')

ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()
