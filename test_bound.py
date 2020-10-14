import numpy as np
import matplotlib.pyplot as plt
import math
from utils import supnorm, normfro
from sklearn.linear_model import LinearRegression

# def getBt(t):
#     a_t_plus_1 = t + 2
#     a_0 = 1.
#     tau_t = 2. / (t + 3)
# 
#     B = (t + 2) * max(-math.log(1 - tau_t), math.log((t + 2) + 1 - tau_t))
# 
#     return B
# 
# B = [getBt(t) for t in range(1000)]
# sB = np.cumsum(B)
# 
# plt.plot(np.arange(1000), sB)
# plt.show()



np.random.seed(999)
nr = 200
nc = 200
n_iter = 1000
C = np.random.uniform(low=1, high=10, size=(nr, nc))
C = (C + C.T) / 2
r = np.random.uniform(low=0.1, high=1, size=(nr, 1))

c = np.random.uniform(low=0.1, high=1, size=(nc, 1))


def conditional_uot(C, r, c, t1=1.0, t2=1.0, n_iter=100):

    X_list = []
    unreg_f_val_list = []
    grad_norm_list = []
    log_sum_r_list = []
    log_sum_c_list = []

    log_r = np.log(r)
    log_c = np.log(c).reshape(1, -1)

    mu = np.sqrt(np.sum(r) * np.sum(c))
    X = np.ones(C.shape) * mu / C.size


    for it in range(n_iter):
        tau = 2 / (it + 3)
        log_sum_r = np.log(X.sum(axis=-1, keepdims=True)) # [r_dim, 1]
        log_sum_c = np.log(X.sum(axis=0, keepdims=True)) # [1, c_dim]
        delta = C + t1 * log_sum_r + t2 * log_sum_c - t1 * log_r - t2 * log_c

        min_indx = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[min_indx] >= 0:
            X = (1 - tau) * X
        else:
            V = np.zeros(delta.shape)
            V[min_indx] = mu
            X = (1 - tau) * X + tau * V
        
        grad_norm_list.append(normfro(delta))
        log_sum_r_list.append(log_sum_r)
        log_sum_c_list.append(log_sum_c)

    info = {
            'grad_norm_list': grad_norm_list,
            'log_sum_r_list': log_sum_r_list,
            'log_sum_c_list': log_sum_c_list,
            }

    return X, info


X, info = conditional_uot(C, r, c, t1=10, t2=10, n_iter=n_iter)
grad_norm = info['grad_norm_list']
log_sum_r = info['log_sum_r_list']
log_sum_c = info['log_sum_c_list']

log_sum_r_diff = [np.max(np.abs(lsr_t1 - lsr_t)) for lsr_t1, lsr_t in zip(log_sum_r[1:], log_sum_r[:-1])]
log_sum_c_diff = [np.max(np.abs(lsc_t1 - lsc_t)) for lsc_t1, lsc_t in zip(log_sum_c[1:], log_sum_c[:-1])]
grad_diff = [supnorm(f_t1 - f_t) for f_t1, f_t in zip(grad_norm[1:], grad_norm[:-1])]
print(grad_diff)
grad_diff_t = grad_diff * np.arange(n_iter-1)
# log_t = - np.log(np.arange(1, n_iter))
# 
# linre = LinearRegression(fit_intercept=True).fit(np.array(log_t), grad_diff)
# print(linre.intercept_, linre.coef_)
# plt.plot(log_t, np.log(grad_diff))
# plt.plot(log_t, log_t * linre.coef_[0,0] + linre.intercept_[0])
# plt.plot(np.arange(n_iter-1), grad_diff_t)
# plt.plot(np.arange(n_iter-1)[400:], grad_diff[400:])
# plt.plot(np.arange(n_iter-1)[400:], 10000 / np.arange(n_iter-1)[400:])
# plt.plot(np.arange(n_iter-1)[400:], grad_diff_t[400:])


# log_sum_diff = np.array(log_sum_r_diff) + np.array(log_sum_c_diff)
# lm = LinearRegression(fit_intercept=True).fit(np.log(np.arange(1, n_iter)).reshape(-1, 1), np.log(log_sum_diff))
# print(lm.coef_)
# print(lm.intercept_)
# plt.plot(np.log(np.arange(n_iter-1)), np.log(log_sum_diff))
# plt.plot(np.log(np.arange(n_iter-1)), np.log(np.arange(n_iter-1)) * lm.coef_ + lm.intercept_)

sum_r_0 = [np.exp(log_sum_r[i][10]) for i in range(n_iter)]
# sum_r_min = [np.min(np.exp(x)) for x in log_sum_r]
# plt.plot(np.arange(n_iter), sum_r_min)
plt.plot(np.arange(n_iter), sum_r_0)
plt.show()
