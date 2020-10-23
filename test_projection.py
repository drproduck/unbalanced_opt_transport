import numpy as np
import utils
import pdb

a = np.random.rand(2, 10)
b = np.random.rand(2)
pdb.set_trace()
b = utils.projection_simplex(a, b, axis=1)
# c = utils.euclidean_proj_simplex(a)

# from sklearn.isotonic import isotonic_regression
# 
# 
# def projection_simplex_isotonic(x, z=1):
#     """
#     Compute argmin_{p : p >= 0 and \sum_i p_i = z} ||p - x||
#     """
#     perm = np.argsort(x)[::-1]
#     x = x[perm]
#     inv_perm = np.zeros(len(x), dtype=np.int)
#     inv_perm[perm] = np.arange(len(x))
#     x[0] -= z
#     dual_sol = isotonic_regression(x, increasing=False)
#     x[0] += z
#     primal_sol = x - dual_sol
#     return primal_sol[inv_perm]
# 
# d = projection_simplex_isotonic(a)

print(a)
print(b)
print(b.sum(1))

# print(d)
# print(d.sum())
