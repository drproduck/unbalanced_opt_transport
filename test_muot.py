import numpy as np
import matplotlib.pyplot as plt
from MUOT import *
import pdb

dims = [100]*3
T = [0.1]*3

C = np.random.uniform(low=1, high=10, size=dims)
C = (C + C.T) / 2

R = []
for d in dims:
    r = np.random.uniform(low=1, high=10, size=(d, 1))
    R.append(r)

info = sinkhorn_muot(C, R, eta=0.01, T=T, n_iter=100, early_stop=False)

fig, ax = plt.subplots(1,3)
ax[0].plot(np.arange(info['stop_iter']+1), info['f_val_list'], label='dual')
ax[1].plot(np.arange(info['stop_iter']+1), info['unreg_f_val_list'], label='unregularized')
ax[2].plot(np.arange(info['stop_iter'])+1, info['err_list'], label='(solution improvement)')
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()
