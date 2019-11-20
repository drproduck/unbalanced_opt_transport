import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb

nr = 100
nc = 100
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=1, high=10, size=(nr, 1))
c = np.random.uniform(low=1, high=10, size=(nc, 1))

u, v, info = sinkhorn_uot(C, r, c, eta=0.01, t1=1.0, t2=1.0, n_iter=1000)

fig, ax = plt.subplots(1,3)
ax[0].plot(np.arange(info['stop_iter']+1), info['f_val_list'], label='dual')
ax[1].plot(np.arange(info['stop_iter']+1), info['unreg_f_val_list'], label='unregularized')
ax[2].plot(np.arange(info['stop_iter'])+1, info['err_list'], label='(solution improvement)')
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()
