from l2_semi_uot import *
from kl_semi_uot import *
import numpy as np
from scipy.spatial.distance import cdist
import pdb
import matplotlib.pyplot as plt
from OT import sinkhorn_ot
import ot

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
b[0, 1] = 1.
b[4, 3] = 1.
b[4, 0] = 1.
b = b.reshape(25, 1)
a = a + 1e-16
b = b + 1e-16

C = cdist(supp, supp, 'euclidean')**2
C = C / C.max()

eta = 1
tau = 100.

X, beta_l2 = fw(C, a, b, tau, duals=True)
X, _, _ = sinkhorn(C, a, b, eta, tau, duals=True)
beta_kl = - X.T.sum(axis=-1, keepdims=True) / b + 1

X_ot, _, beta_ot = sinkhorn_ot(C, a / a.sum(), b / b.sum(), eta, duals=True)
X_pot, log = ot.sinkhorn(a.flatten() / a.sum(), b.flatten() / b.sum(), C, eta, log=True)

beta_pot = eta * np.log(log['v'])
# print(np.abs(X_ot - X_pot).sum())



print(beta_l2.reshape(5, 5))
print(beta_kl.reshape(5, 5))
print(beta_ot.reshape(5, 5))
print(beta_pot.reshape(5, 5))
    

fig, ax = plt.subplots(1, 6)
fig.set_size_inches(9, 3)
ax[0].imshow(a.reshape(5, 5))
ax[0].set_title('a')
ax[1].imshow(b.reshape(5, 5))
ax[1].set_title('b')
ax[2].imshow(beta_l2.reshape(5, 5), cmap='plasma')
ax[2].set_title('l2_pgd')
ax[3].imshow(beta_kl.reshape(5, 5), cmap='plasma', label='kl_uot')
ax[3].set_title('kl_uot')
ax[4].imshow(beta_ot.reshape(5, 5), cmap='plasma')
ax[4].set_title('ot (mine)')
im = ax[5].imshow(beta_pot.reshape(5, 5), cmap='plasma')
ax[5].set_title('pot')

fig.colorbar(im, ax=ax[5])

plt.show()
