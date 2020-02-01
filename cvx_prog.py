import cvxpy as cp
import numpy as np 
import numpy
import matplotlib.pyplot as plt 
from UOT_nostop import * 
from near_linear import sinkhorn_ot as jason
from pavel import sinkhorn_ot as pavel
from prog import exact_ot

np.random.seed(999)
nr = 10
nc = 10
n_eps = 20
C = np.random.uniform(low=1, high=50, size=(nr, nc)).astype(numpy.longdouble)
C = (C + C.T) / 2
# C = C / sum(C) * 15
print('sum C:', C.sum())
a = np.random.uniform(low=0.1, high=1.0, size=nr).astype(numpy.longdouble)
a = a / a.sum() * 1
b = np.random.uniform(low=0.1, high=1.0, size=nc).astype(numpy.longdouble)
b = b / b.sum() * 1
print('sum a:', a.sum(), 'sum b:', b.sum())

tau = 5
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

prob.solve()

print('UOT optimal value:', prob.value)

ot_opt_val = exact_ot(C, a.flatten(), b.flatten())

print('OT optimal value:', ot_opt_val)

def get_eta_k(eps_list):
    alpha = sum(a)
    beta = sum(b)
    # S = (alpha + beta + 1 / np.log(nr)) * (2 * tau + 2)
    S = 0.5 * (alpha + beta) + 0.5 + 0.25 / np.log(nr)
    # T = 4 * ((alpha + beta) * (np.log(alpha + beta) + np.log(nr)) + 1)
    T = 0.5 * (alpha + beta) * (np.log(alpha + beta) - np.log(2) + np.log(nr) - 0.5) + np.log(nr) + 5/2
    
    print('S, T:', S, T)

    eta_list = []
    k_list = []
    for eps in eps_list:
        U = np.max([S + T, 2 * eps, 4 * eps * np.log(nr) / tau, 4 * eps * (alpha+beta) * np.log(nr) / tau])
        eta = eps / U

        R = np.log(a).max() + np.log(b).max() + np.max([np.log(nr), supnorm(C) / eta - np.log(nr)])
        k = (tau * U / eps + 1) * (np.log(8) + np.log(eta) + np.log(R) + np.log(tau) + np.log(tau+1) + 3 * np.log(U) - 3 * np.log(eps))

        eta_list.append(eta)
        k_list.append(k)

    eta_list = np.array(eta_list)
    k_list = np.array(k_list)
    return eta_list, k_list


eps_list = np.linspace(1, 0.05, n_eps)
eta_list, k_list = get_eta_k(eps_list)
print(eps_list)
print(eta_list)

stop_iter_list = []
stop_iter_ot_jason_list = []
opt_iter_ot_jason_list = []

stop_iter_ot_pavel_list = []
opt_iter_ot_pavel_list = []
for eps, eta in zip(eps_list, eta_list):
    print(eps)

    u, v, info = sinkhorn_uot(C, a.reshape(nr,1), b.reshape(nc,1), eta=eta, t1=tau, t2=tau, eps=eps, opt_val=prob.value)
    stop_iter_list.append(info['stop_iter'])

    _, _, info_ot_jason = jason(C, a.reshape(nr, 1), b.reshape(nc, 1), eps=eps, opt_val=ot_opt_val)
    stop_iter_ot_jason_list.append(info_ot_jason['stop_iter'])
    opt_iter_ot_jason_list.append(info_ot_jason['opt_iter'])

    _, _, info_ot_pavel = pavel(C, a.reshape(nr, 1), b.reshape(nc, 1), eps=eps, opt_val=ot_opt_val)
    stop_iter_ot_pavel_list.append(info_ot_pavel['stop_iter'])
    opt_iter_ot_pavel_list.append(info_ot_pavel['opt_iter'])
    
print(opt_iter_ot_jason_list)
print(opt_iter_ot_pavel_list)
print(stop_iter_ot_jason_list)
print(stop_iter_ot_pavel_list)
fig, ax = plt.subplots(1,3)
xs = eps_list
xticks = eps_list[0::2]
yticks = np.arange(0, 70000, 5000)
yrange = (-1000, yticks[-1])


ax[1].set_xticks(xticks)
ax[1].set_yticks(yticks)
ax[1].set_ylim(yrange)
ax[1].set_title('Altschuler OT')
ax[1].plot(xs, opt_iter_ot_jason_list, label=f'max. no. iterations ($k_e$)')
ax[1].plot(xs, stop_iter_ot_jason_list, label='Altschuler bound')
ax[1].set_xlabel('epsilon')
ax[1].set_ylabel('k (iterations)')
ax[1].invert_xaxis()
ax[1].legend(fontsize=15)

ax[2].set_xticks(xticks)
ax[2].set_yticks(yticks)
ax[2].set_ylim(yrange)
ax[2].set_title('Dvurechensky OT')
ax[2].plot(xs, opt_iter_ot_pavel_list, label=f'max. no. iterations ($k_e$)')
ax[2].plot(xs, stop_iter_ot_pavel_list, label='Dvurechensky bound')
ax[2].set_xlabel('epsilon')
ax[2].set_ylabel('k (iterations)')
ax[2].invert_xaxis()
ax[2].legend(fontsize=15)

ax[0].set_xticks(xticks)
ax[0].set_yticks(yticks)
ax[0].set_ylim(yrange)
ax[0].set_title('UOT')
ax[0].plot(xs, stop_iter_list, label=f'max. no. iterations ($k_e$)')
ax[0].set_xlabel('epsilon')
ax[0].set_ylabel('k (iterations)')
k_list = np.array(k_list)
# k_list = stop_iter_list[0] / k_list[0] * k_list
ax[0].plot(xs, k_list, label='Our bound ($k_f$)')
ax[0].invert_xaxis()
ax[0].legend(fontsize=15)

# inv_eps = 1 / eps_list
# inv_eps = stop_iter_list[0] / inv_eps[0] * inv_eps

# inv_eps_2 = 1 / eps_list**2
# inv_eps_2 = stop_iter_list[0] / inv_eps_2[0] * inv_eps_2

# inv_eps_log = 1 / eps_list * np.log(1 / eps_list)
# inv_eps_log = stop_iter_list[0] / inv_eps_log[0] * inv_eps_log

# ax.plot(xs, inv_eps, label='$1 / \epsilon$')
# ax.plot(xs, inv_eps_2, label='$1 / \epsilon^{2}$')
# ax.plot(xs, inv_eps_log, label='$1 / \epsilon * \log (1 / \epsilon)$')


# ax.set_xticklabels(list([f"{x:.3f}" for x in np.linspace(1, 0.1, 10)]))
# plt.xticks(list([f"{x:.3f}" for x in np.linspace(1, 0.1, 10)]))
# print(list([f"{x:.3f}" for x in np.linspace(1, 0.1, 10)]))


plt.show()

