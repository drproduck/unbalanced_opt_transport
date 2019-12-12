import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb

nr = 100
nc = 100
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

u, v, info = sinkhorn_uot(C, r, c, eta=0.01, t1=10, t2=10, n_iter=10000)


f_primal_val = np.array(info['f_primal_val_list'])
f_ratio = f_primal_val[1:] / f_primal_val[:-1] - 1

fig, ax = plt.subplots(1)

ax.plot(np.arange(info['stop_iter']), f_ratio)
ax.legend()

print(r.sum(), c.sum())
plt.show()
