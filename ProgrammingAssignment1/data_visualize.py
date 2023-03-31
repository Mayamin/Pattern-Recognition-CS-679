


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
df = np.array(pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_A.csv', sep=',', header=None))
# print(df.values)

# plt.plot(range(5))
plt.xlim(-5, 15)
plt.ylim(-5, 15)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

x = df[:, [0, 1]]
y = df[:, -1].astype(int)




plt.scatter(x[:,0][y==0], x[:,1][y==0], s=3, c='r')
plt.scatter(x[:,0][y==1], x[:,1][y==1], s=3, c='b')
plt.savefig('dataset_A_plot.pdf', dpi=300)

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.show()


df_2 = np.array(pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_B.csv', sep=',', header=None))
# print(df.values)

plt.xlim(-10, 20)
plt.ylim(-10, 20)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')


x_2 = df_2[:, [0, 1]]
y_2 = df_2[:, -1].astype(int)




plt.scatter(x_2[:,0][y_2==0], x_2[:,1][y_2==0], s=3, c='r')
plt.scatter(x_2[:,0][y_2==1], x_2[:,1][y_2==1], s=3, c='b')
plt.savefig('dataset_B_plot.pdf', dpi=300)
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.show()