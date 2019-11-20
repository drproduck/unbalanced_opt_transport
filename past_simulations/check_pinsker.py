"""
Pinsker fails for non-probability measure!
"""
import numpy as np
import matplotlib.pyplot as plt

def get_KL(P,Q):
    log_ratio = np.log(P) - np.log(Q)
    return np.sum(P * log_ratio - P + Q)
    # return np.sum(P * log_ratio)

def norm1(P, Q):
    return np.sum(np.abs(P - Q))**2

a = np.array([[1],[2],[3]])
a = a / a.sum()
b = np.array([[1/6+0.1],[2/6+0.2],[3/6+0.3]])
print(get_KL(a,b))
print(get_KL(a,b) / norm1(a,b))

# ratios = []
# for i in range(100):
#     P = np.random.uniform(low=1, high=10, size=(2, 1))
#     P = P / P.sum()
#     Q = np.random.uniform(low=1, high=10, size=(2, 1))
#     Q = Q / Q.sum()
#     ratios.append(get_KL(P, Q) / norm1(P, Q))
# 
# print(np.min(ratios))
# plt.plot(np.arange(100), ratios)
# plt.show()
