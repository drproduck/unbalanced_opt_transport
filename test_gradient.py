from l2_uot import gd
import numpy as np
from scipy.spatial.distance import cdist
import pdb
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix


xx = np.arange(5)
yy = np.arange(5)
XX, YY = np.meshgrid(xx, yy, indexing='ij')
supp = np.concatenate((XX.reshape(-1,1), YY.reshape(-1,1)), axis=1)

a = np.zeros((5, 5))
a[4,4] = 1.
a[0, 2] = 1.
a = a.reshape(25, 1)
b = np.zeros((5, 5))
b[0, 0] = 1.
b[4, 2] = 1.
b[4, 0] = 1.
b = b.reshape(25, 1)
# a = np.random.rand(25, 1)
# a[a < 0.8] = 0
# b = np.random.rand(25, 1)
# b[b < 0.8] = 0

C = cdist(supp, supp, 'euclidean')**2

tau = 1.
gamma = 0.1
bs = []

for i in range(10):
    X, _, _ =  gd(C, a, b, tau, 10)
    grad = tau * (b - X.T.sum(axis=-1, keepdims=True))
    b = b - gamma * grad
    b[b < 0] = 0
    bs.append(b)
    

fig, ax = plt.subplots(2, 10)
fig.set_size_inches(20, 4)
ax[0, 0].imshow(a.reshape(5,5))
for i in range(10):
    ax[1, i].imshow(bs[i].reshape(5,5))
    print(a.sum(), bs[i].sum())
print(coo_matrix(X))
plt.show()
