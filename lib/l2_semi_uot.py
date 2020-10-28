import utils
import numpy as np
from utils import projection_simplex
from math import sqrt
import matplotlib.pyplot as plt
import pdb
from copy import copy
import cvxpy as cp
import time


def norm2sq(X):
    return np.sum(X**2)


def fval(C, a, b, tau, X):
    r, c = C.shape
    return np.sum(C * X) + tau / 2. * norm2sq(X - b.T)


def fdual(C, a, b, tau, X):
    beta = tau * (b - X.T.sum(axis=-1, keepdims=True))
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
    Z = X - (0.9 / L) * G
    P = projection_simplex(Z, a, axis=1)
    # print(f'X=X}, G={G}, Z={Z}, P={P}')
    return P


def a_pgd(C, a, b, tau, n_iter=100, X=None, duals=False):
    if X is None:
        X = np.random.rand(*C.shape) + 1e-2
        X = projection_simplex(X, a, axis=1)
    t = 1.
    Y = X
    for i in range(n_iter):
        G = grad(C, b, Y, tau)
        XX = prox(Y, G, tau, a)
        tt = (1 + sqrt(1 + 4 * t**2)) / 2
        Y = XX + ((t - 1) / tt) * (XX - X)
        
        t = tt
        X = XX

    if duals:
        beta = tau * (b - X.T.sum(axis=-1, keepdims=True))
        return X, beta
    else:
        return X

def pgd(C, a, b, tau, gamma=0.1, n_iter=100, X=None, duals=False):

    if X is None:
        X = np.random.rand(*C.shape)
        X = projection_simplex(X, a, axis=1)

    for t in range(n_iter):
        G = grad(C, b, X, tau)
        X = X - gamma * G
        X = projection_simplex(X, a, axis=1)

    if duals:
        beta = tau * (b - X.T.sum(axis=-1, keepdims=True))
        return X, beta
    else:
        return X


def fw(C, a, b, tau, n_iter=100, X=None, duals=False):
    r, c = C.shape

    if X is None:
        X = np.random.rand(*C.shape)
        X = X / X.sum(axis=-1, keepdims=True)
        X = a * X # normalize

    for t in range(n_iter):
        gamma = 2 / (t + 2)
        G = grad(C, b, X, tau)
        idx = np.argmin(G, axis=-1)
        X = (1 - gamma) * X
        X[np.arange(r), idx] = X[np.arange(r), idx] + gamma * a.flatten()

    if duals:
        beta = tau * (b - X.T.sum(axis=-1, keepdims=True))
        return X, beta
    else:
        return X


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
    a = np.random.rand(200,1)
    b = np.random.rand(4000,1)
    C = np.random.rand(200, 4000)
    # C = (C + C.T) / 2
    tau = 1.
    n_iter = 5

    t_s = time.time()
    X = a_pgd(C, a, b, tau, n_iter=n_iter)
    print(time.time() - t_s)
    fs = fval(C, a, b, tau, X)
    print(fs)

    t_s = time.time()
    X = pgd(C, a, b, tau, gamma=0.9/tau, n_iter=n_iter)
    print(time.time() - t_s)
    fs = fval(C, a, b, tau, X)
    print(fs)

    t_s = time.time()
    X = fw(C, a, b, tau, n_iter=100)
    print(time.time() - t_s)
    fs = fval(C, a, b, tau, X)
    print(fs)

    # ex_res, ex_X = exact(C, a, b, tau)

    # print(f'exact: f={ex_res:.3f}, X={ex_X}')
    # print(ex_res)
