import numpy as np
import matplotlib.pyplot as plt

def sinkhorn_log(a, b, C, eta, tau1, tau2, n_iter=100):

    N = [len(a), len(b)]
    H1 = np.ones((N[0], 1))
    H2 = np.ones((N[1], 1))

    def lse(A):
        return np.log(np.sum(np.exp(A), 1))

    def get_M(u, v, C):
        return (- C + u @ H2.T + H1 @ v.T) / eta

    def get_entropy(P):
        logP = np.log(P + 1e-20)
        return -1 * np.sum(P * (logP - 1))

    def get_KL(P, Q):
        log_ratio = np.log(P / Q + 1e-20) 
        return np.sum(P * log_ratio - P + Q)

    def get_KLD(P, Q):
        return np.sum(Q * (np.exp(-P) - 1))

# def KLd(
    def dotp(x, y):
        return np.sum(x * y)

    def norm1(x):
        return np.sum(np.abs(x))

    err = []
    Wprimal_list = []
    Wdual_list = []

    u = np.zeros((N[0], 1), dtype=np.float64)
    v = np.zeros((N[1], 1), dtype=np.float64)

    lam1 = tau1 / (tau1 + eta)
    lam2 = tau2 / (tau2 + eta)

    for i in range(n_iter):
        if i % 2 == 0:
            u1 = np.copy(u)
            u = lam1 * eta * np.log(a) - lam1 * eta * lse(get_M(u, v, C)).reshape(-1, 1) + lam1 * u
            err.append(norm1(u - u1))
        else:
            v1 = np.copy(v)
            v = lam2 * eta * np.log(b) - lam2 * eta * lse(get_M(u, v, C).T).reshape(-1, 1) + lam2 * v
            err.append(norm1(v - v1))

        P = np.exp(get_M(u, v, C))

        Wprimal = dotp(C, P) - eta * get_entropy(P) + tau1 * get_KL(P.sum(axis=1).reshape(-1, 1), a) + tau2 * get_KL(P.sum(axis=0).reshape(-1, 1), b)
        Wdual = - tau1 * get_KLD(u / tau1, a) - tau2 * get_KLD(v / tau2, b) - eta * np.sum(P)
        Wprimal_list.append(Wprimal)
        Wdual_list.append(Wdual)

    return u, v, P, Wprimal_list, Wdual_list, err


n = 2
n_iter = 20
eta=10.0
tau1=1.0
tau2=1.0
# a = np.random.uniform(size=(n, 1))
# b = np.random.uniform(size=(n, 1))
# C = np.random.uniform(low=1, high=10, size=(n, n))
# C = (C + C.T) / 2
a = np.array([[1.0],[3.0]])
b = np.array([[2.0],[4.0]])
C = np.array([[8.0, 6.0],[6.0, 8.0]])

u, v, P, Wprimal, Wdual, err = sinkhorn_log(a, b, C, eta=eta, tau1=tau1, tau2=tau2, n_iter=n_iter)
fig, ax =  plt.subplots(1, 3)

# ax[0].plot(np.arange(n_iter), Wprimal)
# ax[0].set_title('primal')
# ax[1].plot(np.arange(n_iter), Wdual)
# ax[1].set_title('dual')
# ax[2].plot(np.arange(n_iter), err)
# ax[2].set_title('grad norm err')

ax[0].plot(np.arange(n_iter/2), Wprimal[1::2])
ax[0].set_title('primal')
ax[1].plot(np.arange(n_iter/2), Wdual[1::2])
ax[1].set_title('dual')
ax[2].plot(np.arange(n_iter/2), err[1::2])
ax[2].set_title('grad norm err')

print(Wprimal)
plt.show()
