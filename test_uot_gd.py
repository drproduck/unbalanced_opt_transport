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
n_iter = 1000
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

start = time()
X_sh, info_sh = sinkhorn_uot(C, r, c, eta=0.1, t1=10, t2=10, n_iter=n_iter, early_stop=False)
print(np.argmax(X_sh, axis=-1))
print('sinkhorn time elapsed:', time() - start)

start = time()
X_u, info_u = grad_descent_unregularized_uot(C, r, c, t1=10, t2=10, n_iter=n_iter, alpha=1e-3, beta=1e-6, linesearch=False)
print(np.argmax(X_u, axis=-1))
print('gradient descent unregularized time elapsed:', time() - start)

fig, ax = plt.subplots(2)

# converge = info['f_val_list'][-1]
# min_val = np.min(info['f_val_list'])
# ax[0].plot(np.arange(info['stop_iter']+1), info['f_val_list'], label=f'dual, converge={converge:.3f}, min={min_val:.3f}')
# 
# converge = info['f_primal_val_list'][-1]
# min_val = np.min(info['f_primal_val_list'])
# ax[1].plot(np.arange(info['stop_iter']+1), info['f_primal_val_list'], label=f'primal, converge={converge:.3f}, min={min_val:.3f}')
# 
# converge = info['unreg_f_val_list'][-1]
# min_val = np.min(info['unreg_f_val_list'])
# ax[2].plot(np.arange(info['stop_iter']+1), info['unreg_f_val_list'], label=f'unregularized, converge={converge:.3f}, min={min_val:.3f}')
# 
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()

# u_norm = []
# v_norm = []
# for i in range(info['stop_iter']):
#     u_norm.append(supnorm(info['u_list'][i] - info['u_list'][-1]))
#     v_norm.append(supnorm(info['v_list'][i] - info['v_list'][-1]))
# 
# ax.plot(u_norm)
# ax.plot(v_norm)
# ax.set_title('$||u^t - u^*||_{\infty}$')
# plt.show()


# sinkhorn
min_primal_val = np.min(info_sh['f_primal_val_list'])
min_unreg_val = np.min(info_sh['unreg_f_val_list'])
ax[0].plot(np.arange(n_iter+1), info_sh['unreg_f_val_list'], label=f'min={min_unreg_val:.3f}')
# ax[0].plot(np.arange(n_iter+1), info_sh['f_val_list'], label=f'dual')
ax[0].legend()
ax[0].set_title('sinkhorn')


# unregularized
min_unreg_val = np.min(info_u['unreg_f_val_list'])
ax[1].plot(np.arange(n_iter+1), info_u['unreg_f_val_list'], label=f'min={min_unreg_val:.3f}')
ax[1].legend()
ax[1].set_title('gradient descent')

plt.show()
