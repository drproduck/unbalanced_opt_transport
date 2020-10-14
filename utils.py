import numpy as np
import pdb

def get_entropy(P):
    logP = np.log(P + 1e-20)
    return -1 * np.sum(logP * P - P)

def get_KL(P, Q):
    log_ratio = np.log(P + 1e-20) - np.log(Q + 1e-20)
    return np.sum(P * log_ratio - P + Q)

def dotp(x, y):
    return np.sum(x * y)

def norm1(X):
    return np.sum(np.abs(X))

def supnorm(X):
    return np.max(np.abs(X))

def normfro(X):
    return np.sqrt(np.sum(X**2))

def round_rc(F, r, c):
    """
    matrix rounding. Taken from "Near-linear time approximation algorithms for OT via Sinkhorn iterations"
    """
    rF = F.sum(axis=1).reshape(-1, 1)
    x = np.minimum(r / rF, 1)
    FF = F * x.reshape(-1, 1)
    cF = FF.sum(axis=0).reshape(-1, 1)
    y = np.minimum(c / cF, 1)
    FFF = FF * y.reshape(1, -1)

    err_r = r - FFF.sum(axis=1).reshape(-1, 1)
    err_c = c - FFF.sum(axis=0).reshape(-1, 1)
    G = FFF + err_r @ err_c.T / norm1(err_r)

    return G

class BaseGradDescent():
    def __init__(self, lr):
        self.lr = lr
        

    def update(self):
        raise NotImplementedError


class VanillaGradDescent(BaseGradDescent):
    def __init__(self, lr):
        super().__init__(lr)


    def update(self, x, delta):
        return x - self.lr * delta


class NesterovGradDescent(BaseGradDescent):
    def __init__(self, lr):
        super().__init__(lr)
        self.lmd = 0
        self.y_old = None

    def update(self, x, delta):
        lmd_1 = (1 + np.sqrt(1 + 4 * self.lmd ** 2)) / 2
        lmd_2 = (1 + np.sqrt(1 + 4 * lmd_1 ** 2)) / 2
        gamma = (1 - lmd_1) / lmd_2
        print(gamma)
        self.lmd = lmd_1

        y = x - self.lr * delta
        if self.y_old is None:
            x = (1 - gamma) * y + gamma * x
        else:
            x = (1 - gamma) * y + gamma * self.y_old
        self.y_old = y

        return x


class BacktrackGradDescent(BaseGradDescent):
    def __init__(self, lr, beta, eval_func, update_func):
        super().__init__(lr)
        self.beta = beta
        self.eval_func = eval_func
        self.update_func = update_func


    def update(self, x_old, delta, f_old):
        alpha = self.lr
        x_new = self.update_func(x_old, delta)
        f_new = self.eval_func(x_new)
        while (not np.isfinite(f_new)) or (not f_new <= f_old - alpha * self.beta * np.sum(delta**2)):
            alpha /= 2
            x_new = self.update_func(x_old, delta)
            f_new = self.eval_func(x_new)

        return x_new, f_new

""" Module to compute projections on the positive simplex or the L1-ball
A positive simplex is a set X = { \mathbf{x} | \sum_i x_i = s, x_i \geq 0 }
The (unit) L1-ball is the set X = { \mathbf{x} | || x ||_1 \leq 1 }
Adrien Gaidon - INRIA - 2011
"""


import numpy as np


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / rho
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()
