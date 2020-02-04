# import cvxpy as cp
import numpy as np 
import matplotlib.pyplot as plt 
from utils import *
# from UOT import * 
from UOT_torch import *
from copy import copy
from mnist import _load_mnist
from prog import exact_uot
from time import time


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

def get_eta_k(eps, a, b, tau, nr):
    alpha = sum(a)
    beta = sum(b)
    # S = (alpha + beta + 1 / np.log(nr)) * (2 * tau + 2)
    S = 0.5 * (alpha + beta) + 0.5 + 0.25 / np.log(nr)
    # T = 4 * ((alpha + beta) * (np.log(alpha + beta) + np.log(nr)) + 1)
    T = 0.5 * (alpha + beta) * (np.log(alpha + beta) - np.log(2) + np.log(nr) - 0.5) + np.log(nr) + 5/2

    print('S, T:', S, T)

    U = np.max([S + T, 2 * eps, 4 * eps * np.log(nr) / tau, 4 * eps * (alpha+beta) * np.log(nr) / tau])
    eta = eps / U

    def supnorm(X):
        return np.max(np.abs(X))

    R = np.log(a).max() + np.log(b).max() + np.max([np.log(nr), supnorm(C) / eta - np.log(nr)])
    k = (tau * U / eps + 1) * (np.log(8) + np.log(eta) + np.log(R) + np.log(tau) + np.log(tau+1) + 3 * np.log(U) - 3 * np.log(eps))

    return eta[0], k[0]
    
C = get_L1_C(28).astype(np.float64)
print(C)
print(C.sum())

x, y = _load_mnist('.', split_type='train', download=True)
pairs = ((3, 5), (20562, 12428), (2564, 12380), (48485, 7605), (26428, 42698), (6152, 25061), (13168, 7506), (40816, 39370), (846, 16727), (31169, 7144))

#3 5
#sum C: 11458944.0 sum a: 66.93428175 sum b: 115.62950225
#float64 float64 float64
#time elapsed: 460.77280616760254
#exact val: 321.83947656441
#20562 12428
#sum C: 11458944.0 sum a: 68.36784349999999 sum b: 69.031928
#float64 float64 float64
#time elapsed: 487.2219512462616
#exact val: 167.64555073518932
#2564 12380
#sum C: 11458944.0 sum a: 55.41083125 sum b: 79.83269625
#float64 float64 float64
#time elapsed: 550.395977973938
#exact val: 207.3502556064093
#48485 7605
#sum C: 11458944.0 sum a: 156.02792975 sum b: 74.31317100000001
#float64 float64 float64
#time elapsed: 524.784610748291
#exact val: 423.35617236431176
#26428 42698
#sum C: 11458944.0 sum a: 56.62177174999999 sum b: 49.86397025
#float64 float64 float64
#time elapsed: 456.55393266677856
#exact val: 117.52384257739209
#6152 25061
#sum C: 11458944.0 sum a: 147.52010325 sum b: 64.3053795
#float64 float64 float64
#time elapsed: 539.6158065795898
#exact val: 459.297428039594
#13168 7506
#sum C: 11458944.0 sum a: 95.094414 sum b: 105.36390925
#float64 float64 float64
#time elapsed: 546.8977901935577
#exact val: 138.082535273433
#40816 39370
#sum C: 11458944.0 sum a: 93.641251 sum b: 67.95769725
#float64 float64 float64
#time elapsed: 509.2965066432953
#exact val: 256.3855206718765
#846 16727
#sum C: 11458944.0 sum a: 112.719378 sum b: 81.30143325
#float64 float64 float64
#time elapsed: 548.7409589290619
#exact val: 221.13453306061183
#31169 7144
#sum C: 11458944.0 sum a: 76.3053745 sum b: 108.66078325000001
#float64 float64 float64
#time elapsed: 527.7265672683716
#exact val: 348.8398254732447

for (id1, id2) in pairs:
	print(id1, id2)
	a = x[id1].astype(np.float64)
	b = x[id2].astype(np.float64)
	# fig, ax = plt.subplots(1, 2)
	# ax[0].imshow(a.reshape(28, 28))
	# ax[0].set_title(y[id1])
	# ax[1].imshow(b.reshape(28, 28))
	# ax[1].set_title(y[id2])
	a = a.reshape(-1, 1)
	b = b.reshape(-1, 1)
	a[a == 0] = 1e-6
	b[b == 0] = 1e-6

	tau = 10
	print('sum C:', C.sum(), 'sum a:', a.sum(), 'sum b:', b.sum())
	print(C.dtype, a.dtype, b.dtype)

	start = time()
	uot_opt_val = exact_uot(C, a.flatten(), b.flatten(), tau)
	print('time elapsed:', time() - start)
	print('exact val:', uot_opt_val)
	# time = 252s, opt_val = 305.794

# start = time()
# eps = 1.0
# eta, k = get_eta_k(eps, a, b, tau, 784)
# print('eta:', eta, 'k:', k)
# eta = 0.001
# u, v, info = sinkhorn_uot(C, a, b, eta=eta, t1=tau, t2=tau, n_iter=10000000, eps=eps, opt_val=305.794)
# print('time elapsed:', time() - start)
# # print(info['unreg_f_val_list'])
#     
# print('approx val:', info['unreg_f_val_list'][-1])
# print(info['stop_iter'])

# plt.show()

