import numpy as np
import matplotlib.pyplot as plt
from MUOT import *
import pdb

n = 3
dims = [2]*3
T = [10]*3

C = np.random.uniform(low=1, high=50, size=dims)
C = (C + C.T) / 2

R = []
for d in dims:
    r = np.random.uniform(low=0.1, high=1, size=(d, 1))
    R.append(r)

info = sinkhorn_muot(C, R, eta=0.1, T=T, n_iter=500, early_stop=False)

# inf_norms = {}
# for i in range(n):
#     inf_norms[i] = []
# 
# for i in range(info['stop_iter']):
#     for j in range(n):
#         inf_norms[j].append(supnorm(info['U_all'][i][j] - info['U_all'][-1][j]))

fig, ax = plt.subplots(1,3)
min_primal = np.min(info['f_primal_val_list'])
min_dual = np.min(info['f_dual_val_list'])
min_unregularized = np.min(info['unreg_f_val_list'])
conv_primal = info['f_primal_val_list'][-1]
conv_dual = info['f_dual_val_list'][-1]
conv_unregularized = info['unreg_f_val_list'][-1]
ax[0].plot(np.arange(info['stop_iter']+1), info['f_primal_val_list'], label=f'primal, {min_primal:.3f}, {conv_primal:.3f}')
ax[2].plot(np.arange(info['stop_iter']+1), info['f_dual_val_list'], label=f'dual, {min_dual:.3f}, {conv_dual:.3f}')
ax[1].plot(np.arange(info['stop_iter']+1), info['unreg_f_val_list'], label=f'unregularized, {min_unregularized:.3f}, {conv_unregularized:.3f}')
ax[0].legend()
ax[1].legend()
ax[2].legend()

# for j in range(n):
#     ax.plot(np.arange(info['stop_iter'])+1, inf_norms[j], label='marginal i={}'.format(j))

# ax.set_title('$||u^t_i - u^*_i||$')

# U = np.concatenate(info['U_all'], axis=-1).transpose(0, 2, 1) # [3, 10000, 2]
# for i in range(3):
#     for j in range(info['stop_iter']):
#         U[i, j] = U[i, j, :] - U[i, -1, :]
# print(U[0, -1, :])
# for i in range(3):
#     ax[i].scatter(U[i][:,0], U[i][:,1], s=5)
#     ax[i].legend()
plt.show()
