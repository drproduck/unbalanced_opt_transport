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
n_iter = 5000
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

start = time()
X_sh, info_sh = sinkhorn_uot(C, r, c, eta=0.1, t1=10, t2=10, n_iter=n_iter, early_stop=False)
print(np.argmax(X_sh, axis=-1))
print('sinkhorn time elapsed:', time() - start)

start = time()
X_gd, info_gd = grad_descent_uot(C, r, c, eta=0.001, t1=10, t2=10, n_iter=n_iter, gamma=1e-3)
print(np.argmax(X_gd, axis=-1))
print('gradient descent time elapsed:', time() - start)

# start = time()
# X_gd, info_gd = grad_descent_uot_with_linesearch(C, r, c, eta=0.1, t1=10, t2=10, n_iter=n_iter, alpha=0.001, beta=0.5)
# print(np.argmax(X_gd, axis=-1))
# print('gradient descent with linesearch time elapsed:', time() - start)

start = time()
X_gdr, info_gdr = grad_descent_exp_uot(C, r, c, eta=0.001, t1=10, t2=10, n_iter=n_iter, gamma=0.01)
print(np.argmax(X_gdr, axis=-1))
# print('gradient descent reparameterized time elapsed:', time() - start)
# print(info_gdr['f_primal_val_list'])


start = time()
X_gpu, info_gpu = gpu.grad_descent_exp_uot(C, r, c, eta=0.1, t1=10, t2=10, n_iter=n_iter, gamma=1.)
print(np.argmax(X_gpu, axis=-1))
print('gradient descent GPU time elapsed:', time() - start)


start = time()
X_u, info_u = grad_descent_unregularized_uot(C, r, c, t1=10, t2=10, n_iter=n_iter, alpha=1e-3, beta=1e-6, linesearch=False)
print(np.argmax(X_u, axis=-1))
print('gradient descent unregularized time elapsed:', time() - start)

fig, ax = plt.subplots(2,5)

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
ax[0,0].plot(np.arange(n_iter+1), info_sh['f_primal_val_list'], label=f'primal, min={min_primal_val:.3f}')
ax[0,0].plot(np.arange(n_iter+1), info_sh['unreg_f_val_list'], label=f'unreg, min={min_unreg_val:.3f}')
# ax[0].plot(np.arange(n_iter+1), info_sh['f_val_list'], label=f'dual')
ax[0,0].legend()
ax[0,0].set_title('sinkhorn')


# GD
min_primal_val = np.min(info_gd['f_primal_val_list'])
min_unreg_val = np.min(info_gd['unreg_f_val_list'])
ax[0,1].plot(np.arange(n_iter+1), info_gd['f_primal_val_list'], label=f'primal, min={min_primal_val:.3f}')
ax[0,1].plot(np.arange(n_iter+1), info_gd['unreg_f_val_list'], label=f'unreg, min={min_unreg_val:.3f}')
ax[0,1].legend()
ax[0,1].set_title('gradient descent')

min_grad_norm_val = np.min(info_gd['grad_norm_list'])
ax[1,1].plot(np.arange(n_iter), info_gd['grad_norm_list'], label=f'grad norm, min={min_grad_norm_val:.3f}')
ax[1,1].legend()


# GD exp
min_primal_val = np.min(info_gdr['f_primal_val_list'])
min_unreg_val = np.min(info_gdr['unreg_f_val_list'])
ax[0,2].plot(np.arange(n_iter+1), info_gdr['f_primal_val_list'], label=f'primal, min={min_primal_val:.3f}')
ax[0,2].plot(np.arange(n_iter+1), info_gdr['unreg_f_val_list'], label=f'unreg, min={min_unreg_val:.3f}')
ax[0,2].legend()
ax[0,2].set_title('gradient descent reparam')

min_grad_norm_val = np.min(info_gdr['grad_norm_list'])
ax[1,2].plot(np.arange(n_iter), info_gdr['grad_norm_list'], label=f'grad norm, min={min_grad_norm_val:.3f}')
ax[1,2].legend()


# gpu
min_primal_val = np.min(info_gpu['f_primal_val_list'])
min_unreg_val = np.min(info_gpu['unreg_f_val_list'])
ax[0,3].plot(np.arange(n_iter), info_gpu['f_primal_val_list'], label=f'primal, min={min_primal_val:.3f}')
ax[0,3].plot(np.arange(n_iter), info_gpu['unreg_f_val_list'], label=f'unreg, min={min_unreg_val:.3f}')
ax[0,3].legend()
ax[0,3].set_title('gradient descent reparam gpu')

min_grad_norm_val = np.min(info_gpu['grad_norm_list'])
ax[1,3].plot(np.arange(n_iter), info_gpu['grad_norm_list'], label=f'grad norm, min={min_grad_norm_val:.3f}')
ax[1,3].legend()


# unregularized
min_unreg_val = np.min(info_u['unreg_f_val_list'])
ax[0,4].plot(np.arange(n_iter+1), info_u['unreg_f_val_list'], label=f'unreg, min={min_unreg_val:.3f}')
ax[0,4].legend()
ax[0,4].set_title('gradient descent unregularized')

min_grad_norm_val = np.min(info_u['grad_norm_list'])
ax[1,4].plot(np.arange(n_iter), info_u['grad_norm_list'], label=f'grad norm, min={min_grad_norm_val:.3f}')
ax[1,4].legend()
plt.show()
