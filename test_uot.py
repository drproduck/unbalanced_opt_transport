import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb
from time import time
from utils import *
from gpu import UOT_torch as gpu

np.random.seed(999)
nr = 200
nc = 200
n_iter = 1000
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

start = time()
X_sh, info_sh = sinkhorn_uot(C, r, c, eta=0.1, t1=10, t2=10, n_iter=100, early_stop=False)


plt.plot(np.arange(info_sh['stop_iter']+1), info_sh['f_val_list'])
plt.plot(np.arange(info_sh['stop_iter']+1), info_sh['f_primal_val_list'])
print('sinkhorn time elapsed:', time() - start)
plt.show()
