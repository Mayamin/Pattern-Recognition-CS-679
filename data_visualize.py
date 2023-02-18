


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
df = np.array(pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_A_part1.csv', sep=',', header=None))
# print(df.values)

x = df[:, [0, 1]]
y = df[:, -1].astype(int)

plt.scatter(x[:,0][y==0], x[:,1][y==0], s=3, c='r')
plt.scatter(x[:,0][y==1], x[:,1][y==1], s=3, c='b')
plt.savefig('figure_1.pdf', dpi=300)
plt.show()


