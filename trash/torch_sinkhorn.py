import torch
from torch import nn
import matplotlib.pyplot as plt
def norm2(X, sq=True):
    if sq:
        return torch.sum(X**2, dim=-1)
    else:
        return torch.sum(X**2, dim=-1).sqrt()

def norm1(X, Y):
    return torch.sum(torch.abs(X - Y))
def sinkhorn_loss(Xb, Mu, eta, n_iter, r, c):
    """not log_stable
    :arg Xb: data batch (row-based)
    :arg Mu: mean vector (row_based)
    :arg r: first marginal (cores. Xb)
    :arg c: second marginal (cores. Xb)
    Using L2 distance
    """

    d1 = Xb.shape[0]
    d2 = Mu.shape[0]
    Xb_norm = norm2(Xb, sq=True).reshape(-1, 1)
    Mu_norm = norm2(Mu, sq=True).reshape(-1, 1)
    C = Xb_norm + Mu_norm.t() - 2 * Xb @ Mu.t()
    print(C)

    def get_B(x, y):
        return (-C + x + y.T) / eta

    x = torch.zeros(size=(d1, 1)).type(Mu.dtype)
    y = torch.zeros(size=(d2, 1)).type(Mu.dtype)

    norm1_list = []
    val_list = []
    for i in range(n_iter):
        if i % 2 == 0:
        B = torch.exp( (-C + x + y.t()) / eta)
        x = x + torch.log(r) - torch.log(B.sum(dim=1).reshape(-1, 1))
        B = torch.exp( (-C + x + y.t()) / eta)
        y = y + torch.log(c) - torch.log(B.sum(dim=0).reshape(-1, 1))

        B = torch.exp( (-C + x + y.t()) / eta)
        norm1_list.append(norm1(B.sum(dim=1).reshape(-1,1), r) + norm1(B.sum(dim=0).reshape(-1,1), c))
        val_list.append(torch.sum(C * B))

    return val_list, norm1_list

Xb = torch.Tensor(size=(10, 2)).uniform_(1, 10)
Mu = torch.Tensor(size=(10, 2)).uniform_(1, 10)

r = torch.Tensor(size=(10, 1)).uniform_()
r = r / r.sum()

c = torch.Tensor(size=(10, 1)).uniform_()
c = c / c.sum()


val_list, norm1_list = sinkhorn_loss(Xb, Mu, eta=1, n_iter=1000, r=r, c=c)
plt.plot(val_list, label='val')
plt.plot(norm1_list, label='norm1')
plt.legend()
plt.show()
