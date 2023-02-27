
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# reading dataset A
df_1 = np.array(pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_A.csv', sep=',', header=None))
# df_2 = pd.read_csv("/home/mraha/Desktop/myproject/src/dataset_A.csv")

# reading datset B
df_3 = np.array(pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_B.csv', sep=',', header=None))


# for dataset A parsing the feature values
# getting x feature values from pandas dataframe
# getting y feature values from pandas dataframe
x_A = df_1[:, [0]]
y_A =df_1[:, [1]]
# total number of samples
n = 200000
# array to store prediction values
prediction_A = []

# given values for mean
u0x_A = u0y_A = 1
u1x_A = u1y_A = 4

#things needed for calculating misclassification rate for each class separately

#variables to store total number of correct predictions
correct_predictions_class0_A = correct_predictions_class1_A = 0
incorrect_predictions_class0_A = incorrect_predictions_class1_A = 0

#variables to store total number of incorrect predictions
correct_predictions_class0_B = correct_predictions_class1_B = 0
incorrect_predictions_class0_B = incorrect_predictions_class1_B = 0

# for dataset B parsing the feature values
# getting x feature values from pandas dataframe
# getting y feature values from pandas dataframe
x_B = df_3[:, [0]]
y_B = df_3[:, [1]]
# total number of samples

# array to store prediction values
prediction_B = []

# given values for mean
u0x_B = u0y_B = 1
u1x_B = u1y_B = 4

# total count of predictions per class for dataset A and B
class0_prediction_number_A = class1_prediction_number_A = 0
class0_prediction_number_B = class1_prediction_number_B = 0


def Euclidean_Distance(x, y, ux, uy):

	# calculating number to square in next step
    x = x - ux 
    y = y - uy
  
    # calculating Euclidean distance
    dist = pow(x, 2) + pow(y, 2)      
    dist = math.sqrt(dist)                

    return dist


for i in range(n):
  

        dist_class0_A = Euclidean_Distance(x_A[i], y_A[i], u0x_A, u0y_A) 
        dist_class1_A = Euclidean_Distance(x_A[i], y_A[i], u1x_A, u1y_A) 


        dist_class0_B = Euclidean_Distance(x_B[i], y_B[i], u0x_B, u0y_B) 
        dist_class1_B = Euclidean_Distance(x_B[i], y_B[i], u1x_B, u1y_B) 

    
        if dist_class0_A < dist_class1_A:

            prediction_A.append(0)

        else:
       
            prediction_A.append(1)  

        if dist_class0_B < dist_class1_B:

            prediction_B.append(0)

        else:
       
            prediction_B.append(1)  

        # sample code incase we want to separate features that have a 50 percent possibilty of classification in each class  
        # elif dist_class0 > dist_class1:
        #     # prediction = np.append(prediction, [0])
        #     # prediction[i] = 1
        #     prediction.append(1)

        # else:
        #     # prediction[i] = -1
        #     prediction.append(-1)


# classEqual_prediction_number = 0

for i in range(len(prediction_A)):
    # print(prediction[i])

    if prediction_A[i] == 0:
        class0_prediction_number_A += 1
    else:
        class1_prediction_number_A += 1

    if prediction_B[i] == 0:
        class0_prediction_number_B += 1
    else:
        class1_prediction_number_B += 1

    # sample code incase we want to separate features that have a 50 percent possibilty of classification in each class    
    # elif prediction[i] == 1:
    #     class1_prediction_number += 1
    # else:
    #     classEqual_prediction_number += 1

# checking total number of predictions = total data points
print ("class0_prediction_number_A ",class0_prediction_number_A, " class1_prediction_number_A ", class1_prediction_number_A, " total prediction for dataset_A", class0_prediction_number_A+class1_prediction_number_A)
print ("class0_prediction_number_B ",class0_prediction_number_B, " class1_prediction_number_B ", class1_prediction_number_B, " total prediction for dataset_B", class0_prediction_number_B+class1_prediction_number_B)
# print (classEqual_prediction_number)



# misclassification rate for each class 0 for dataset A and B 
label_A = df_1[:, [2]]
label_B = df_3[:, [2]]

misclassifcation_rate_class0_A = misclassifcation_rate_class1_A = 0
misclassifcation_rate_class0_B = misclassifcation_rate_class1_B = 0

# print(label_A[0])

for i in range (60000):

    #for dataset A class 0

    if (label_A[i]-prediction_A[i]) == 0:

        correct_predictions_class0_A += 1
    else:

        incorrect_predictions_class0_A += 1

    #for dataset B class 0

    if (label_B[i]-prediction_B[i]) == 0:

        correct_predictions_class0_B += 1
    else:

        incorrect_predictions_class0_B += 1

#calculating misclassification error rate for class 0 in dataset A and B        

misclassifcation_rate_class0_A = (incorrect_predictions_class0_A / 60000) * 100
misclassifcation_rate_class0_B = (incorrect_predictions_class0_B / 60000) * 100

feature_number_class1_A = feature_number_class1_B = 0
# misclassification rate for each class 1 for dataset A and B 

for i in range(60000, 200000):

    #for dataset A class 1

    if (label_A[i]-prediction_A[feature_number_class1_A]) == 0:

        correct_predictions_class1_A += 1
        feature_number_class1_A += 1
    else:

        incorrect_predictions_class1_A += 1
        feature_number_class1_A += 1

    #for dataset B class 1

    if (label_B[i]-prediction_B[feature_number_class1_B ]) == 0:

        correct_predictions_class1_B += 1
        feature_number_class1_B += 1

    else:

        incorrect_predictions_class1_B += 1
        feature_number_class1_B += 1

misclassifcation_rate_class1_A = (incorrect_predictions_class1_A / 140000) * 100
misclassifcation_rate_class1_B = (incorrect_predictions_class1_B / 140000) * 100

print ("misclassifcation_rate_class1_A ", misclassifcation_rate_class1_A , "misclassifcation_rate_class1_B ", misclassifcation_rate_class1_B)





