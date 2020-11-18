import numpy as np
import matplotlib.pyplot as plt

n = 1000
s = 0

ss = []
for i in range(1, n):
    s = s + np.log(i) / i**2
    ss.append(s)


plt.plot(np.arange(1, n), ss / np.log(np.arange(1, n)))
plt.plot(np.arange(1, n), 1 / np.arange(1, n))
plt.show()
