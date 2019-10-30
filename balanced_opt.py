import numpy as np
import matplotlib.pyplot as plt
# regularize constant epsilon

eps = 10
n_pts = 1000
n_iter = 10
r = np.random.uniform(size=(n_pts, 1))
r = r / r.sum()
c = np.random.uniform(size=(n_pts, 1))
c = c / c.sum()

def get_entropy(P):
    logP = np.log(P)
    return -np.sum(P * logP)

A = np.random.uniform(low=1, high=10, size=(n_pts, n_pts))
A = (A + A.T) / 2
K = np.exp(- A / eps)

u = np.ones((n_pts, 1))
v = np.ones((n_pts, 1))
row_diff = []
col_diff = []
tol_diff = []
grad_norm = []
for i in range(1, n_iter+1):
    if i % 2 == 1:
        u = r / (K @ v)
        # print(np.sum(K * u * v.T))
    else:
        v = c / (K.T @ u)
        # print(np.sum(K * u * v.T))

    P = K * u * v.T
    rd = np.sum(np.abs(P.sum(axis=1).reshape(n_pts, 1) - r))
    cd = np.sum(np.abs(P.sum(axis=0).reshape(n_pts, 1) - c))

    row_diff.append(rd)
    col_diff.append(cd)
    tol_diff.append(rd + cd)

    # gradient norm
    Kv = K @ v
    ru = r / u
    # print(ru)
    Du = eps * np.abs(Kv - ru).sum()
    KTu = K.T @ u
    cv = c / v
    Dv = eps * np.abs(KTu - cv).sum()
    grad_norm.append(Du + Dv)
    print(Du + Dv)

fig, ax = plt.subplots(3, 1)
ax[0].plot(np.arange(1, n_iter+1), row_diff, label='||r(P) - r||_1')
ax[0].plot(np.arange(1, n_iter+1), col_diff, label='||c(P) - c||_1')
ax[1].plot(np.arange(1, n_iter+1), tol_diff, label='||r(P) - r||_1 + ||c(P) - c||_1')
ax[2].plot(np.arange(1, n_iter+1), grad_norm, label='||du f(u,v)||_1 + ||dv f(u,v)||_1')
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()
