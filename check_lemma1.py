import numpy as np
import matplotlib.pyplot as plt
from UOT import *
import pdb
from utils import  *

nr = 100
nc = 100
C = np.random.uniform(low=10, high=100, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))

u, v, info = sinkhorn_uot(C, r, c, eta=0.1, t1=10, t2=10, n_iter=1000, early_stop=False)

a_diff_list = []
b_diff_list = []
for i in range(1000):
    u = info['u_list'][i]
    v = info['v_list'][i]
    B = get_B(C, u, v, eta=0.1)
    astar = B.sum(axis=1).reshape(-1, 1)
    bstar = B.sum(axis=0).reshape(-1, 1)
    a_diff = supnorm(np.log(r) - np.log(astar) - u / 10)
    b_diff = supnorm(np.log(c) - np.log(bstar) - v / 10)
    a_diff_list.append(a_diff)
    b_diff_list.append(b_diff)

fig, ax = plt.subplots(1,1)

ax.plot(a_diff_list)
ax.plot(b_diff_list)
print(a_diff_list[-1], b_diff_list[-1])
plt.show()
