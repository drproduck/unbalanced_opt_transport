import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb

nr = 10
nc = 10
fig, ax = plt.subplots(5, 1)
fig.suptitle('t1 * KL(a, r) + t2 * KL(b, c)')

C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

eta_list = np.linspace(0.01, 0.5, 5)

for i, eta in enumerate(eta_list):
    u, v, info = sinkhorn_uot(C, r, c, eta=eta, t1=10, t2=10, n_iter=10000, early_stop=False)
 
    converge = info['kl_list'][-1]
    min_val = np.min(info['kl_list'])
    ax[i].plot(np.arange(10001), info['kl_list'], label=f'eta={eta:.3f}, converge={converge:.3f}, min={min_val:.3f}')
    ax[i].legend()

print(sum(r), sum(c))
plt.show()

