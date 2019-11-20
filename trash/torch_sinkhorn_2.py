import torch
from torch import nn
import matplotlib.pyplot as plt

def norm2(X, sq=True):
    if sq:
        return torch.sum(X**2, dim=-1)
    else:
        return torch.sum(X**2, dim=-1).sqrt()

def norm1(X,Y):
    return torch.sum(torch.abs(X-Y))
def sinkhorn_loss(Xb, Mu, eps, n_iter, m1, m2):
    """not log_stable
    :arg Xb: data batch (row-based)
    :arg Mu: mean vector (row_based)
    :arg m1: first marginal (cores. Xb)
    :arg m2: second marginal (cores. Xb)
    Using L2 distance
    """
    d1 = Xb.shape[0]
    d2 = Mu.shape[0]
    Xb_norm = norm2(Xb, sq=True).reshape(-1, 1)
    Mu_norm = norm2(Mu, sq=True).reshape(-1, 1)
    C = Xb_norm + Mu_norm.t() - 2 * Xb @ Mu.t()
    K = torch.exp(- C / eps)
    print(C)

    norm1_list = []
    val_list = []
    b = torch.ones(size=(d2, 1)).type(Mu.dtype)
    for i in range(n_iter):
        a = m1 / (K @ b)
        b = m2 / (K.t() @ a)
        P = K * a * b.t()
        # print(torch.sum(C * P))
        norm1_list.append(norm1(P.sum(dim=1).reshape(-1,1), m1) + norm1(P.sum(dim=0).reshape(-1,1), m2))
        val_list.append(torch.sum(C * P))
    return val_list, norm1_list


Xb = torch.Tensor(size=(10,2)).uniform_(1, 10)
Mu = torch.Tensor(size=(10,2)).uniform_(1, 10)
m1 = torch.Tensor(size=(10,1)).uniform_(1, 10)
m1 = m1 / m1.sum()
m2 = torch.Tensor(size=(10,1)).uniform_(1, 10)
m2 = m2 / m2.sum()
val_list, norm1_list = sinkhorn_loss(Xb, Mu, eps=0.1, n_iter=100, m1=m1, m2=m2)
plt.plot(norm1_list, label='norm1')
plt.plot(val_list, label='val')
plt.legend()
plt.show()
