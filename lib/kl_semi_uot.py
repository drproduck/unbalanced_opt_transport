from time import time
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import numpy as np
from scipy.special import logsumexp



def get_S(C, u, v, eta):
    K = - C + u + v.T
    return np.exp(K / eta)


def f_primal(C, r, c, u, v, S, eta, tau):
    return np.sum(C * S) + tau * KL(np.sum(S.T, axis=-1, keepdims=True), c) - eta * H(S)


def f_dual(C, r, c, u, v, S, eta, tau):
    return np.sum(r * u) - tau * np.sum(c * np.exp(- v / tau)) - eta * np.sum(S)


def sinkhorn(C, r, c, eta=0.1, tau=10., n_iter=100, duals=False):
    """
    \min_{X, X1 = a} <C, X> + \tau KL(X^T 1, b) - H(X)
    
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg eta: entropic-regularizer
    :arg t2: Kl regularizer
    :n_iter: number of Sinkhorn iterations
    """

    # log_r = np.log(r + 1e-16)
    # log_c = np.log(c + 1e-16)

    log_r = np.log(r)
    log_c = np.log(c)


    # initial solution
    u = np.zeros_like(r)
    v = np.zeros_like(c)
    S = get_S(C, u, v, eta)


    for i in range(n_iter):
#             S = get_S(C, u, v, eta)
#             b = S.sum(dim=0).reshape(-1, 1)
        K = - C + u + v.T
        log_b = logsumexp(K.T / eta, axis=-1, keepdims=True)
        v = (v / eta + log_c - log_b) * (tau * eta / (eta + tau))

        # we end the loop with update of a so that row sum constraint is satisfied.
#             S = get_S(C, u, v, eta)
#             a = S.sum(dim=1).reshape(-1, 1)
        K = - C + u + v.T
        log_a = logsumexp(K / eta, axis=-1, keepdims=True)
        u = (u / eta + log_r - log_a) * eta

        S = get_S(C, u, v, eta)


    if duals:
        return S, u, v
    else:
        return S

if __name__ == '__main__':
    r = torch.rand(10, 1).cuda()
    r = r / r.sum()
    c = torch.rand(20, 1).cuda()
    
    C = torch.rand(10, 20).cuda()
    eta = 0.1
    n_iter = 100
    tau = 1.
    u, v, S, us, vs, Ss = sinkhorn(C, r, c, eta=eta, tau=tau, n_iter=n_iter)

    f_primals = [f_primal(C, r, c, u, v, S, eta, tau).item() for u, v, S in zip(us, vs, Ss)]
    f_duals = [f_dual(C, r, c, u, v, S, eta, tau).item() for u, v, S in zip(us, vs, Ss)]

    print(f_duals)
    print(f_primals)

    plt.plot(np.arange(101), f_duals, label='dual')
    plt.plot(np.arange(101), f_primals, label='primal')
    plt.legend()
    plt.show()

