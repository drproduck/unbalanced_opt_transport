import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb
from time import time
from utils import *
from gpu import UOT_torch as gpu

np.random.seed(0)
nr = 100
nc = 100
n_iter = 10000
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

start = time()
X_sh, info_sh = sinkhorn_uot(C, r, c, eta=0.1, t1=10, t2=10, n_iter=n_iter, early_stop=True)
print(np.argmax(X_sh, axis=-1))
print('sinkhorn time elapsed:', time() - start)


start = time()
X_u, info_u = grad_descent_unregularized_uot(C, r, c, t1=10, t2=10, n_iter=n_iter, alpha=1e-3, beta=1e-6, linesearch=False)
print(np.argmax(X_u, axis=-1))
print('gradient descent unregularized time elapsed:', time() - start)

fig, ax = plt.subplots(2,2)


# sinkhorn
min_primal_val = np.min(info_sh['f_primal_val_list'])
min_unreg_val = np.min(info_sh['unreg_f_val_list'])
# ax[0,0].plot(np.arange(info_sh['stop_iter']+1), info_sh['f_primal_val_list'], label=f'primal, min={min_primal_val:.3f}')
ax[0,0].plot(np.arange(info_sh['stop_iter']+1), info_sh['unreg_f_val_list'], label=f'unreg, min={min_unreg_val:.3f}')
# ax[0].plot(np.arange(n_iter+1), info_sh['f_val_list'], label=f'dual')
ax[0,0].legend()
ax[0,0].set_title('sinkhorn')


# unregularized
min_unreg_val = np.min(info_u['unreg_f_val_list'])
ax[0,1].plot(np.arange(n_iter+1), info_u['unreg_f_val_list'], label=f'unreg, min={min_unreg_val:.3f}')
ax[0,1].legend()
ax[0,1].set_title('gradient descent unregularized')

min_grad_norm_val = np.min(info_u['grad_norm_list'])
ax[1,1].plot(np.arange(n_iter), info_u['grad_norm_list'], label=f'grad norm, min={min_grad_norm_val:.3f}')
ax[1,1].legend()
plt.show()
