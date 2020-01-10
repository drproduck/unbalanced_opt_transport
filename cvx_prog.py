import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from UOT import *

nr = 100
nc = 100
C = np.random.uniform(low=10, high=100, size=(nr, nc))
C = (C + C.T) / 2
a = np.random.uniform(low=0.1, high=1, size=nr)

b = np.random.uniform(low=0.1, high=1, size=nc)

tau = 10
X = cp.Variable((nr,nc), nonneg=True)


row_sums = cp.sum(X, axis=1)
col_sums = cp.sum(X, axis=0)

obj = cp.sum(cp.multiply(X, C))

obj -= tau * cp.sum(cp.entr(row_sums))
obj -= tau * cp.sum(cp.entr(col_sums))

obj -= tau * cp.sum(cp.multiply(row_sums, cp.log(a)))
obj -= tau * cp.sum(cp.multiply(col_sums, cp.log(b)))

obj -= 2 * tau * cp.sum(X)
obj += tau * cp.sum(a) + tau * cp.sum(b)

prob = cp.Problem(cp.Minimize(obj))

prob.solve()

print('optimal value', prob.value)
print(X.value)


u, v, info = sinkhorn_uot(C, a.reshape(nr,1), b.reshape(nc,1), eta=0.1, t1=tau, t2=tau, n_iter=1000, early_stop=False)
print(info['unreg_f_val_list'][-1])


plt.plot(np.arange(1001), info['unreg_f_val_list'], c='blue')
plt.plot([0, 1000], [prob.value, prob.value], c='red')
plt.show()






