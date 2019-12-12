import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb

nr = 10
nc = 10
fig, ax = plt.subplots(10, 1)
fig.suptitle('unregularized function value')

C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

eta_list = np.linspace(0.01, 0.5, 10)

for i, eta in enumerate(eta_list):
    u, v, info = sinkhorn_uot(C, r, c, eta=eta, t1=10, t2=10, n_iter=10000, early_stop=False)
 
    f_converge = info['unreg_f_val_list'][-1]
    f_min = np.min(info['unreg_f_val_list'])
    ax[i].plot(np.arange(10001), info['unreg_f_val_list'], label=f'eta={eta:.3f}, converge={f_converge:.3f}, min={f_min:.3f}')
    ax[i].legend()

print(sum(r), sum(c))
plt.show()

