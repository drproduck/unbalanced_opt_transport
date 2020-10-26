import utils
import numpy as np
from utils import projection_simplex
from math import sqrt
import matplotlib.pyplot as plt
import pdb
from copy import copy
import cvxpy as cp

# v = np.random.rand(3, 5)
# z = np.array([0.2, 0.3, 0.5])
# b = projection_simplex(v, z, axis=1)
# print(b)
# print(b.sum(1))

def norm2sq(X):
    return np.sum(X**2)

def fval(C, a, b, tau, X):
    r, c = C.shape
    return np.sum(C * X) + tau / 2. * norm2sq(X - b.T)

def fdual(C, a, b, tau, X):
    beta = - tau * (X.T.sum(axis=-1, keepdims=True) - b)
    alpha = np.min(C - beta.T, axis=-1, keepdims=True)
    d = np.sum(a * alpha) + np.sum(b * beta) - 1. / (2 * tau) * norm2sq(beta)

    return d


def grad(C, b, X, tau):
    """
    C: [n_a, n_b]
    X: [n_a, n_b]
    a: [n_a, 1]
    b: [n_b, 1]
    """

    r, c = C.shape
    return C + tau * (X - b.T)


def prox(X, G, L, a):
    """
    X: [n_a, n_b]
    G: [n_a, n_b]
    """
    # pdb.set_trace()
    Z = X - (0.9 / L) * G
    P = projection_simplex(Z, a.flatten(), axis=1)
    # print(f'X=X}, G={G}, Z={Z}, P={P}')
    return P


def fista(C, a, b, tau, n_iter=100, X=None):
    h, w = C.shape
    Xs = []
    Gs = []
    if X is None:
        X = np.random.rand(*C.shape) + 1e-2
        X = projection_simplex(X, a.flatten(), axis=1)
    t = 1.
    Y = copy(X)
    Xs.append(X)
    for t in range(n_iter):
        G = grad(C, b, Y, tau)
        XX = prox(Y, G, tau, a)
        tt = (1 + sqrt(1 + 4 * t**2)) / 2
        Y = XX + ((t - 1) / tt) * (XX - X)
        
        t = tt
        X = copy(XX)
        
        Xs.append(X)
        Gs.append(G)

    return X, Xs, Gs

def gd(C, a, b, tau, gamma=0.1, n_iter=100, X=None):
    Xs = []
    Gs = []
    if X is None:
        X = np.random.rand(*C.shape)
        X = projection_simplex(X, a.flatten(), axis=1)
    Xs.append(X)

    for t in range(n_iter):
        G = grad(C, b, X, tau)
        X = X - gamma * G
        X = projection_simplex(X, a.flatten(), axis=1)
        Xs.append(X)
        Gs.append(G)

    return X, Xs, Gs


def exact(C, a, b, tau):
    r, c = C.shape
    X = cp.Variable(C.shape, nonneg=True)
    constraints = [cp.sum(X, axis=1, keepdims=True) == a]
    obj = cp.sum(cp.multiply(X, C))
    obj += tau / 2 * cp.sum_squares(X - np.ones((r, 1)) @ b.T)
    prob = cp.Problem(cp.Minimize(obj), constraints)

    result = prob.solve(solver='SCS')
    return result, X.value

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.rand(10,1)
    b = np.random.rand(20,1)
    C = np.random.rand(10,20)
    # C = (C + C.T) / 2
    tau = 1.
    n_iter = 100
    X, Xs, Gs = fista(C, a, b, tau, n_iter=n_iter)
    fs = [fval(C, a, b, tau, X) for X in Xs]
    fds = [fdual(C, a, b, tau, X) for X in Xs]
    # print(f'fista: f={fs[-1]:.3f}, X={X}')
    print(fs[-1])
    gnorms = [sqrt(norm2sq(G)) for G in Gs]
    fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(np.arange(n_iter+1), fs)
    ax[0,0].plot(np.arange(n_iter+1), fds)
    ax[0,0].set_title('nesterov pgd')
    ax[1,0].plot(np.arange(n_iter), gnorms)
    ax[1,0].set_title('grad norm')

    X, Xs, Gs = gd(C, a, b, tau, gamma=0.9/tau, n_iter=n_iter)
    fs = [fval(C, a, b, tau, X) for X in Xs]
    fds = [fdual(C, a, b, tau, X) for X in Xs]
    # print(f'pgd: f={fs[-1]:.3f}, X={X}')
    print(fs[-1])
    gnorms = [sqrt(norm2sq(G)) for G in Gs]
    ax[0,1].plot(np.arange(n_iter+1), fs)
    ax[0,1].plot(np.arange(n_iter+1), fds)
    ax[0,1].set_title('pgd')
    ax[1,1].plot(np.arange(n_iter), gnorms)
    ax[1,1].set_title('grad_norm')

    ex_res, ex_X = exact(C, a, b, tau)

    # print(f'exact: f={ex_res:.3f}, X={ex_X}')
    print(ex_res)
    plt.show()
