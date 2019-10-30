import numpy as np
import matplotlib.pyplot as plt
t1 = 10
t2 = 10
eps = 1

n_pts = 10
# r = np.array([0.3, 0.7]).reshape(2,1)
# c = np.array([0.5, 0.5]).reshape(2,1)
r = np.random.uniform(size=(n_pts, 1))
c = np.random.uniform(size=(n_pts, 1))
A = np.random.randn(n_pts, n_pts)
A = (A + A.T) / 2
K = np.exp(-A / eps)

u = np.ones((n_pts,1))
v = np.ones((n_pts,1))
grad_norm = []
row_diff = []
col_diff = []
tol_diff = []
n_iter = 100
for i in range(1,n_iter):
    if i % 2 == 1:
        u = (r / (K @ v))**(t1 / (t1 + eps))
    else:
        v = (c / (K.T @ u))**(t2 / (t2 + eps))

    P = K * u * v.T

    rd = np.sum(np.abs(P.sum(axis=1).reshape(n_pts, 1) - r))
    # print(P.sum(axis=1))
    row_diff.append(rd)
    cd = np.sum(np.abs(P.sum(axis=0).reshape(n_pts, 1) - c))
    col_diff.append(cd)
    tol_diff.append(rd + cd)
    # compute gradient
    Kv = K @ v
    ur = u**(- eps/t1 - 1) * r
    Du = eps * np.abs(Kv - ur).sum()
    KTu = K.T @ u
    vc = v**(- eps/t2 - 1) * c
    Dv = eps * np.abs(KTu - vc).sum()
   # print(Kv.shape, ur.shape, Du.shape, KTu.shape, vc.shape, Dv.shape)
    grad_norm.append(Du + Dv)

fig, ax = plt.subplots(1,3)
ax[0].plot(np.arange(1,n_iter), grad_norm, label='||du f(u,v)||_1 + ||dv f(u,v)||_1')
ax[0].set_title('gradient norm 1')
ax[0].legend()
ax[1].plot(np.arange(1,n_iter), row_diff, label='||r(P) - r||_1')
ax[1].plot(np.arange(1,n_iter), col_diff, label='||c(P) - c||_1')
ax[1].set_title('row difference and column difference')
ax[1].legend()
ax[2].plot(np.arange(1,n_iter), tol_diff, label='||r(P) - r||_1 + ||c(P) - c||_1')
ax[2].set_title('total difference (objective)')
ax[2].legend()
print(tol_diff)

plt.show()
