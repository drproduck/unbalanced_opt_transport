from utils import *
import numpy as np

x = np.random.rand(10)
y = np.random.rand(10)

print(get_KL(x, y))
print(get_KL(x / x.sum(), y / y.sum()))
