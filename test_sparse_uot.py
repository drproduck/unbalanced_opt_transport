import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb
from time import time
from utils import *
from gpu import UOT_torch as gpu

np.random.seed(0)
nr = 10
nc = 10
n_iter = 10000
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))


start = time()
subprob_args = {'alpha':1e-3, 'beta':1e-6, 'n_iter':100, 'linesearch': False}
X_s, info_s = sparse_uot(C, r, c, t1=10, t2=10, alpha=1e-2, gamma=1, n_iter=n_iter, n_sample=1, subprob_args=subprob_args)
print(np.argmax(X_s, axis=-1))
print('gd sparse uot time elapsed:', time() - start)
print(X_s)

fig, ax = plt.subplots(1)

# sparse
ax.plot(np.arange(n_iter), info_s['rand_f_list'])
plt.show()
