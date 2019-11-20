"""
the entropic-regularized Unbalanced OT landscape.
To be able to plot this, we limit the number of atoms to only 1 for each marginal.
Varying eta, the entropic-regularizer
"""
import numpy as np
import matplotlib.pyplot as plt
n_pts = 1

a = 0.5
b = 0.7

C = 1

def f(u, v, C, a, b, eta=1, t1=1, t2=1):
    K = np.exp(- C / eta)
    B = K * np.exp(u / eta) * np.exp(v / eta)
    obj = eta * B
    kl1 = t1 * np.exp(-u / eta) * a
    kl2 = t2 * np.exp(-v / eta) * b

    return obj + kl1 + kl2

N = 1000
x = np.linspace(-10, 10, N)
y = np.linspace(-10, 10, N)
xv, yv = np.meshgrid(x, y)
xv = xv.flatten().reshape(-1, 1)
yv = yv.flatten().reshape(-1, 1)

fig, ax = plt.subplots(2, 5)
for i, eta in enumerate(np.linspace(1, 5, 10)):
    v = f(xv, yv, C=C, a=a, b=b, eta=eta)
    min_val = np.min(v)
    v = min_val - v
    v[v < -10] = -10

    ax[i//5, i%5].contourf(xv.reshape(N, N), yv.reshape(N, N), v.reshape(N, N))
    ax[i//5, i%5].set_title(f'eta={eta:.3f}')

plt.show()
