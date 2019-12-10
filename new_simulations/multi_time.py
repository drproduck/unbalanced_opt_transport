import numpy as np
import matplotlib.pyplot as plt
from MUOT import *
import pdb
time_list = []

for n_dim in range(2, 20):
    dims = [2]*n_dim
    T = [1.0]*n_dim

    C = np.random.uniform(low=1, high=10, size=dims)
    C = (C + C.T) / 2

    R = []
    for d in dims:
        r = np.random.uniform(low=1, high=10, size=(d, 1))
        R.append(r)

    info = sinkhorn_muot(C, R, eta=0.01, T=T, n_iter=200, early_stop=True)
    time_list.append(sum(info['time_per_iter_list']))

fig, ax = plt.subplots(1,1)
ax.plot(range(2, 20), time_list)
plt.show()
