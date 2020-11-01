from l2_semi_uot import *
from kl_semi_uot import sinkhorn as kl_sinkhorn
import numpy as np
from scipy.spatial.distance import cdist
import pdb
import matplotlib.pyplot as plt
from OT import sinkhorn_ot
import ot
from utils import *

from scipy.sparse import coo_matrix


def im_text(ax, data):
    size = 5

    # The normal figure
    im = ax.imshow(data, origin='lower', interpolation='None', cmap='viridis')

    [x_start, x_end, y_start, y_end] = im.get_extent()

    # Add the text
    jump_x = (x_end - x_start) / (2.0 * size)
    jump_y = (y_end - y_start) / (2.0 * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = str(f'{data[y_index, x_index]:.1f}')
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center')

#    fig.colorbar(im)




a = np.zeros((5, 5))
a[4,4] = 1.
a[0, 2] = 1.
a = a.reshape(25, 1)
# supp_a = np.array([[0,2], [4,4]])
# a = np.array([1., 1.]).reshape(-1, 1)
b = np.zeros((5, 5))
b[0, 1] = 2.
b[4, 3] = 3.
b[4, 0] = 4.
b = b.reshape(25, 1)
a = a + 1e-16
b = b + 1e-16

xx = np.arange(5)
yy = np.arange(5)
XX, YY = np.meshgrid(xx, yy, indexing='ij')
supp = np.concatenate((XX.reshape(-1,1), YY.reshape(-1,1)), axis=1)

C = cdist(supp, supp, 'euclidean')**2

eta = 0.001
tau = 0.1

X, beta_l2 = fw(C, a, b, tau, duals=True)
beta_l2 = beta_l2 / tau
X, _, _ = kl_sinkhorn(C, a, b, eta, tau, duals=True)
beta_kl = - X.T.sum(axis=-1, keepdims=True) / b + 1

anorm = a / a.sum()
bnorm = b / b.sum()
X_ot, _, beta_ot = sinkhorn_ot(C, anorm, bnorm, eta, duals=True)
beta_ot = beta_ot / norm1(b) - np.sum(beta_ot*b) / norm1(b)**2

X_pot, log = ot.sinkhorn(anorm.flatten(), bnorm.flatten(), C, reg=eta, log=True)

beta_pot = eta * np.log(log['v'])
beta_pot = beta_pot.reshape(-1, 1)
beta_pot = beta_pot / norm1(b) - np.sum(beta_pot*b) / norm1(b)**2


beta_l1 = np.zeros(b.shape)
beta_l1[b > a] = 1.
beta_l1[b < a] = -1.


# print(beta_l2.reshape(5, 5))
# print(beta_kl.reshape(5, 5))
# print(beta_ot.reshape(5, 5))
# print(beta_pot.reshape(5, 5))
    
def linf(x):
    return np.max(np.abs(x))


def update(b, beta, gamma=1.):
    print(b.shape, beta.shape)
    res = b - gamma * beta
    res[res < 0] = 0
    return res

fig, ax = plt.subplots(1, 7)
fig.set_size_inches(9, 3)
ax[0].imshow(a.reshape(5, 5))
ax[0].set_title('a')
im_text(ax[0], a.reshape(5,5))

ax[1].imshow(b.reshape(5, 5))
ax[1].set_title('b')
im_text(ax[1], b.reshape(5,5))

# ax[2].imshow(beta_l2.reshape(5, 5), cmap='plasma')
im_text(ax[2], beta_l2.reshape(5, 5))
ax[2].set_title(f'l2_pgd_{linf(beta_l2):.3f}')

# ax[3].imshow(beta_kl.reshape(5, 5), cmap='plasma')
im_text(ax[3], beta_kl.reshape(5, 5))
ax[3].set_title(f'kl_uot_{linf(beta_kl):.3f}')

# ax[4].imshow(beta_ot.reshape(5, 5), cmap='plasma')
im_text(ax[4], beta_ot.reshape(5, 5))
ax[4].set_title(f'ot_(mine)_{linf(beta_ot):.3f}')

# ax[5].imshow(beta_pot.reshape(5, 5), cmap='plasma')
im_text(ax[5], beta_pot.reshape(5, 5))
ax[5].set_title(f'pot_{linf(beta_pot):.3f}')

# ax[6].imshow(beta_l1.reshape(5, 5), cmap='plasma')
im_text(ax[6], beta_l1.reshape(5, 5))
ax[6].set_title(f'l1_{linf(beta_l1):.3f}')

# fig.colorbar(im, ax=ax[5])

plt.show()
