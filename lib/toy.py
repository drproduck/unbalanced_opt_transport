import numpy as np
from l2_semi_uot import pgd, fval, fw
from kl_semi_uot import sinkhorn
import ot
import pdb

a = np.array([0.25, 0.75]).reshape(-1, 1)
b = np.array([0.75, 0.25]).reshape(-1, 1)
print(a)
print(b)

C = np.array([[0, 1],[1, 0]])

tau = 100000.
# X = pgd(C,a,b,tau=tau,gamma=1./tau)

X = fw(C, a, b, tau=tau)
print(X)
f = fval(C, a, b, tau, X)
print(f)

X = sinkhorn(C, a, b, eta=0.01, tau=tau)
print(X)
f = fval(C, a, b, tau, X)
print(f)

X = ot.sinkhorn(a.flatten(), b.flatten(), C, reg=0.01)
print(X)
f = fval(C, a, b, tau, X)
print(f)


# print(X)
# 
