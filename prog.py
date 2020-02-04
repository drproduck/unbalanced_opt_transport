import numpy as np
import numpy
import cvxpy as cp
from utils import *


def exact_ot(C, a, b):
    nr = len(a)
    nc = len(b)
    X = cp.Variable((nr,nc), nonneg=True)

    row_sums = cp.sum(X, axis=1)
    col_sums = cp.sum(X, axis=0)

    obj = cp.sum(cp.multiply(X, C))

    prob = cp.Problem(cp.Minimize(obj), [row_sums == a, col_sums == b])

    prob.solve()

    x = X.value

    # print(norm1(x.sum(axis=1) - a))
    # print(norm1(x.sum(axis=0) - b))

    return prob.value


def exact_uot(C, a, b, tau):
    nr = len(a)
    nc = len(b)
    X = cp.Variable((nr,nc), nonneg=True)

    row_sums = cp.sum(X, axis=1)
    col_sums = cp.sum(X, axis=0)

    obj = cp.sum(cp.multiply(X, C))

    obj -= tau * cp.sum(cp.entr(row_sums))
    obj -= tau * cp.sum(cp.entr(col_sums))

    obj -= tau * cp.sum(cp.multiply(row_sums, cp.log(a)))
    obj -= tau * cp.sum(cp.multiply(col_sums, cp.log(b)))

    obj -= 2 * tau * cp.sum(X)
    obj += tau * cp.sum(a) + tau * cp.sum(b)

    prob = cp.Problem(cp.Minimize(obj))

    prob.solve(solver='SCS')

    # print('UOT optimal value:', prob.value)

    return prob.value


