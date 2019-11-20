from utils import *
from UOT import sinkhorn_uot
import matplotlib.pyplot as plt

C = np.random.randn(2,2)

C = (C + C.T) / 2
r = np.random.uniform(low=1, high=10, size=(2,1))
c = np.random.uniform(low=1, high=10, size=(2,1))

u_list, v_list, f_val_list = sinkhorn_uot(C, r, c, eta=0.01, n_iter=1000)
print(u_list)

u_list = np.array(u_list)
v_list = np.array(v_list)


u_list = u_list - u_list[-1]
print(u_list.shape)
v_list = v_list - v_list[-1]

fig, ax = plt.subplots(2,2)
ax[0][0].plot(np.arange(1001), u_list[:,0], label='u[0]')
ax[0][1].plot(np.arange(1001), u_list[:,1], label='u[1]')
ax[1][0].plot(np.arange(1001), u_list[:,0], label='v[0]')
ax[1][1].plot(np.arange(1001), u_list[:,1], label='v[1]')
ax[0][0].legend()
ax[0][1].legend()
ax[1][0].legend()
ax[1][1].legend()

plt.show()
