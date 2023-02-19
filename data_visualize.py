


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
df = np.array(pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_A.csv', sep=',', header=None))
# print(df.values)

x = df[:, [0, 1]]
y = df[:, -1].astype(int)

plt.scatter(x[:,0][y==0], x[:,1][y==0], s=3, c='r')
plt.scatter(x[:,0][y==1], x[:,1][y==1], s=3, c='b')
plt.savefig('dataset_A_plot.pdf', dpi=300)
plt.show()


df_2 = np.array(pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_B.csv', sep=',', header=None))
# print(df.values)

x_2 = df_2[:, [0, 1]]
y_2 = df_2[:, -1].astype(int)

plt.scatter(x_2[:,0][y_2==0], x_2[:,1][y_2==0], s=3, c='r')
plt.scatter(x_2[:,0][y_2==1], x_2[:,1][y_2==1], s=3, c='b')
plt.savefig('dataset_B_plot.pdf', dpi=300)
plt.show()