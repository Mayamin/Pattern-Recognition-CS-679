
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


df = np.array(pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_A.csv', sep=',', header=None))
u0x = u0y = 1
u1x = u1y = 4


def Euclidean_Distance(x, y, ux, uy):

	# calculating number to square in next step
    x = x - ux 
    y = y - uy
  

    # calculating Euclidean distance
    dist = pow(x, 2) + pow(y, 2)      
    dist = math.sqrt(dist)                

    return dist


x = df[:, [0]]
y =df[:, [1]]

n = 200000

# prediction = np.empty(n)
prediction = []
print(len(prediction))
# prediction = prediction[:].astype(int)

# print (x)


# print (y)

# type(x_2[0])  <class 'numpy.ndarray'>

# print( x[0] )

df_2 = pd.read_csv("/home/mraha/Desktop/myproject/src/dataset_A.csv")
# df_2[0:1, [2]] = 0
# predicted_class = df_2[0:1, [2]]
# print(np.array(df_2))

# print(predicted_class[0])

print(len(x))
print(len(y))

for i in range(n):
        # print(i)
        # # finidng which mean is closer to datapoints
        # print(x[i])
        # print(y[i])


        dist_class0 = Euclidean_Distance(x[i], y[i], u0x, u0y) 
        dist_class1 = Euclidean_Distance(x[i], y[i], u1x, u1y) 

        # print(dist_class0)
        # print(dist_class1)

        if dist_class0 < dist_class1:

            # prediction = np.append(prediction, [0])
            # prediction[i] = 0
            prediction.append(0)

        elif dist_class0 > dist_class1:
            # prediction = np.append(prediction, [0])
            # prediction[i] = 1
            prediction.append(1)

        else:
            # prediction[i] = -1
            prediction.append(-1)


class0_prediction_number = 0
class1_prediction_number = 0
classEqual_prediction_number = 0

for i in range(len(prediction)):
    # print(prediction[i])

    if prediction[i] == 0:
        class0_prediction_number += 1
    elif prediction[i] == 1:
        class1_prediction_number += 1
    else:
        classEqual_prediction_number += 1

print (class0_prediction_number)
print (class1_prediction_number)
print (classEqual_prediction_number)






