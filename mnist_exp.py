import cvxpy as cp
import numpy as np 
import matplotlib.pyplot as plt 
from UOT import * 
from copy import copy
from mnist import _load_mnist

def repeat_newdim(x, n_repeat, newdim):
     x = np.expand_dims(x, axis=newdim)
     x = np.repeat(x, n_repeat, newdim)
     return x


def get_L1_C(dim):
    xv, yv = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
    vv1 = np.concatenate((xv.reshape(-1,1), yv.reshape(-1, 1)), axis=1)

    vv2 = copy(vv1)
    vv1 = repeat_newdim(vv1, dim**2, 1)
    vv2 = repeat_newdim(vv2, dim**2, 0)
    # print(vv1.shape, vv2.shape)
    C = vv1 - vv2
    C = np.abs(C).sum(axis=-1)
    return C
    
C = get_L1_C(28).astype(np.float32)
print(C)
print(C.sum())

x, y = _load_mnist('.', split_type='train', download=True)
id1 = 3
id2 = 5
a = x[id1]
b = x[id2]
fig, ax = plt.subplots(1, 2)
ax[0].imshow(a.reshape(28, 28))
ax[0].set_title(y[id1])
ax[1].imshow(b.reshape(28, 28))
ax[1].set_title(y[id2])
a = a.reshape(-1, 1) + 0.01
b = b.reshape(-1, 1) + 0.01

tau = 10
print('sum C:', C.sum(), 'sum a:', a.sum(), 'sum b:', b.sum())
print(C.dtype, a.dtype, b.dtype)

u, v, info = sinkhorn_uot(C, a, b, eta=0.001, t1=tau, t2=tau, n_iter=1000, early_stop=True, eps=None, opt_val=None)
    
print(info['unreg_f_val_list'][-1])

# plt.show()

