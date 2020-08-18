import numpy as np
import matplotlib.pyplot as plt
from UOT import sinkhorn_uot

np.random.seed(42)

n = 5000


def batch_eudist(A, B, squared=True, clip=0.):
    """
    A: shape = [..., m, d]
    B: shape = [..., n, d]
    """
    m = A.shape[-2]
    n = B.shape[-2]
    A_norm = np.sum(A**2, axis=-1, keepdims=True) # [..., m, 1]
    B_norm = np.sum(B**2, axis=-1, keepdims=True) # [..., n, 1]
    A_xp = np.expand_dims(A, -2) # [..., m, 1, d]
    B_xp = np.expand_dims(B, -3) # [..., 1, n, d]
    AdotB = np.sum(A_xp * B_xp, axis=-1) # [..., m, n]

    res_sq = A_norm + B_norm.T - 2 * AdotB
    if squared:
        return res_sq
    else:
        res = res_sq.clone()
        pos_idx = res > 0
        res[pos_idx] = torch.sqrt(res_sq[pos_idx])
        return res

# These are optimal settings derived from above investigation.
scale = 0.01
tau1 = 1
tau2 = 1
X_max = 1

alpha = 2 * scale
beta = 4 * scale
a = np.ones(shape=(n,)) / n * alpha
b = np.ones(shape=(n,)) / n * beta

X = np.random.rand(n, 3) * X_max
C = batch_eudist(X, X, squared=True)

epsilon = 1.0
tau = tau1

# Calculate quantities
S = 1 / 2 * (alpha + beta) + 1 / 2 + 1 / (4 * np.log(n))
T = 1 / 2 * (alpha + beta) * (np.log((alpha + beta) / 2) + 2 * np.log(n) - 1) + np.log(n) + 5 / 2
U = max(S + T, epsilon, 4 * epsilon * np.log(n) / tau, 4 * epsilon * (alpha + beta) * np.log(n) / tau)

eta = epsilon / U


X, info = sinkhorn_uot(C, a, b, eta=eta, t1=tau, t2=tau, n_iter=100)


plt.plot(np.arange(1001), info['unreg_f_val_list'])
plt.show()
