import numpy as np
import matplotlib.pyplot as plt
from GUOT import *
from UOT import *
import pdb

nr = 500
nc = 500
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=1, high=10, size=(nr, 1))
c = np.random.uniform(low=1, high=10, size=(nc, 1))

n_iter = 10000
eta = 0.01

# uot
_, _, info_uot = sinkhorn_uot(C, r, c, eta=eta, t1=1.0, t2=1.0, n_iter=n_iter, early_stop=False)

# guot
_, _, info_guot = sinkhorn_guot(C, r, c, eta=eta, t1=1.0, t2=1.0, n_iter=n_iter)

fig, ax = plt.subplots(2,2)
ax[0, 0].set_title('uot dual')
ax[0, 0].plot(np.arange(n_iter+1), info_uot['f_val_list'], label='uot, dual')
ax[0, 1].set_title('uot unregularized')
ax[0, 1].plot(np.arange(n_iter+1), info_uot['unreg_f_val_list'], label='uot, unregularized')
ax[0, 0].legend()
ax[0, 1].legend()

ax[1, 0].set_title('gout dual')
ax[1, 0].plot(np.arange(n_iter+1), info_guot['f_val_list'], label='guot, dual')
ax[1, 1].set_title('gout unregularized')
ax[1, 1].plot(np.arange(n_iter+1), info_guot['unreg_f_val_list'], label='guot, unregularized')
ax[1, 0].legend()
ax[1, 1].legend()
plt.show()
