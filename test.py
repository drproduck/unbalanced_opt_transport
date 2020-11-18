import numpy as np

C = np.array([[1,2],[3,4]])
u = np.array([[1],[2]])
v = np.array([[2],[3]])

print(C.sum(axis=1).reshape(-1, 1))

K = -C + u + v.T
# print('C', C)
# print()
# print('u', u)
# print()
# print('v', v)
# print()
# print('K', K)
