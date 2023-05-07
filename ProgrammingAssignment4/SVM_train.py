import numpy as NP
from libsvm.svmutil import *
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import csv

#read train img_coeff
path_tr01 = Path("GenderDataRowOrder/16_20/trPCA_01.txt")
dataframe1 = pd.read_csv(path_tr01 )
dataframe1.to_csv('trPCA_01.csv', index = None)
tr_01 = [[] for i in range(134)]
X_train = []
with open('trPCA_01.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        # print(row)
        #splitting the elements of first column of each row
        row_arr = row[0].split()
        #converting values in each row to float
        row_arr = np.array([ float(i) for i in row_arr ])
        for i in range( len(row_arr)):
            tr_01[i].append(row_arr[i])
    
    X_train = np.transpose(np.array(tr_01))[:, :30]
    # print("X_train", X_train)
print("X_train.shape ",X_train.shape )

#read train img_label
path_Ttr01 = Path("GenderDataRowOrder/16_20/TtrPCA_01.txt")
dataframe2 = pd.read_csv(path_Ttr01 )
dataframe2.to_csv('TtrPCA_01.csv', index = None)

with open('TtrPCA_01.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    first_row = next(csv_reader)
    y_train = np.array(first_row[0].split())
# print("train_labels",y_train) # print("train_labels[0].type",type(y_train[0])) ////str
print("train_labels.shape",y_train.shape)


#read test img coeff
path_ts01 = Path("GenderDataRowOrder/16_20/tsPCA_01.txt")
dataframe3 = pd.read_csv(path_ts01 )
dataframe3.to_csv('tsPCA_01.csv', index = None)
ts_01 = [[] for i in range(134)]
X_test = []
with open('tsPCA_01.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        # print(row)
        #splitting the elements of first column of each row
        row_arr = row[0].split()
        #converting values in each row to float
        row_arr = np.array([ float(i) for i in row_arr ])
        for i in range( len(row_arr)):
            ts_01[i].append(row_arr[i])
    X_test = np.transpose(np.array(ts_01))[:, :30]
    # print("X_test", X_test)
print("X_test.shape ",X_test.shape )

#read test img_label
path_Tts01 = Path("GenderDataRowOrder/16_20/TtsPCA_01.txt")
dataframe4 = pd.read_csv(path_Tts01 )
dataframe4.to_csv('TtsPCA_01.csv', index = None)
with open('TtsPCA_01.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    first_row = next(csv_reader)
    y_test = np.array(first_row[0].split())
print(" y_test.shape, type(y_test[0]), y_test ", y_test.shape,type(y_test[0]), y_test)


#trg d = degree of polynomial = 1,2,3, C = regularization parameter
poly = SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
poly_pred = poly.predict(X_test)
poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))




