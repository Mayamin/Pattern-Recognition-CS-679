
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


df = np.array(pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_A.csv', sep=',', header=None))
u0x = u0y = u1x = u1y = 1


def Euclidean_Distance(x, y, ux, uy):

	# calculating number to square in next step
    x = x - ux 
    y = y - uy
  

    # calculating Euclidean distance
    dist = pow(x, 2) + pow(y, 2)      
    dist = math.sqrt(dist)                

    return dist

	# f = open('/home/mraha/Desktop/myproject/src/dataset_A.csv', 'r')
    # reader = csv.reader(f)
    # mylist = list(reader)
    # f.close()
    # mylist[1][3] = 'X'
    # my_new_list = open('mylist.csv', 'w', newline = '')
    # csv_writer = csv.writer(my_new_list)
    # csv_writer.writerows(mylist)
    # my_new_list.close()





x = df[:, [0]]
y =df[:, [1]]

prediction = np

print (x[199999])

print (y[199999])

# type(x_2[0])  <class 'numpy.ndarray'>

# print( type(x_2[0]) )

df_2 = pd.read_csv("/home/mraha/Desktop/myproject/src/dataset_A.csv")
df_2[0:1, [2]] = 0
predicted_class = df_2[0:1, [2]]
print(np.array(df_2))

print(predicted_class[0])

# for i in x_2:
# 	# finidng which mean is closer to datapoints
# 	dist_class0 = Euclidean_Distance(x[0], y[0], u0x, u0y) 
# 	dist_class1 = Euclidean_Distance(x[0], y[0], u1x, u1y) 

# 	if dist_class0 > dist_class1:




