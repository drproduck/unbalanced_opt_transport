import numpy as np
import matplotlib.pyplot as plt
"""
Stablized implementation using log domain
"""
def softmin(X, eps, axis=1):
    """
    compute softmin for each row (column) of matrix X
    """
    E = np.exp(- X / eps)
    E = E.sum(axis=axis).reshape(-1, 1)
    return -eps * np.log(E)


eps = 1
n_pts = 100
n_iter = 100
col_ones = np.ones((n_pts, 1))
row_ones = np.ones((1, n_pts))
r = np.random.uniform(size=(n_pts, 1))
r = r / r.sum()
log_r = np.log(r)
c = np.random.uniform(size=(n_pts, 1))
c = c / c.sum()
log_c = np.log(c)

C = np.random.randn(n_pts, n_pts)
C = (C + C.T) / 2
K = np.exp(- C / eps)

f = np.ones((n_pts, 1)) / n_pts
f = np.log(f)
g = np.ones((n_pts, 1)) / n_pts
g = np.log(g)

row_diff = []
col_diff = []
tol_diff = []
grad_norm = []
for i in range(1, n_iter+1):
    if i % 2 == 1:
        f = softmin(C - f @ row_ones - g.T @ col_ones, eps=eps, axis=1) - f + eps * log_r
    else:
        g = softmin(C - f @ row_ones - g.T @ col_ones, eps=eps, axis=0) - g + eps * log_c

    u = np.exp(f / eps)
    v = np.exp(g / eps)
    P = K * u * v.T
    print(np.sum(P))
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

fig, ax = plt.subplots(1, 3)
ax[0].plot(np.arange(1, n_iter+1), row_diff)
ax[1].plot(np.arange(1, n_iter+1), col_diff)
ax[1].plot(np.arange(1, n_iter+1), tol_diff)
ax[2].plot(np.arange(1, n_iter+1), grad_norm)
plt.show()
