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
