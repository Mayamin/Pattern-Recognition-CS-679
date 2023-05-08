
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import csv
from svm_fold import *
from itertools import permutations

total_fold_number = 3
poly_degrees = 3

error_poly_rbf_f123 = [
    [0] * 19,
    [0] * 19,
    [0] * 19
]
with open('error_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in error_poly_rbf_f123:
        writer.writerow(row)
df = pd.read_csv('error_data.csv', header=None)

for i in range(total_fold_number):
    img_no = 134
    fold_no = i+1
    data= svm_fold_x(img_no,fold_no)

    #polynomial kernel parameter value d = 1,2,3
    for d in range (poly_degrees):  
        #SVM train_test on poly kernel d= 1  
        poly = SVC(kernel='poly', degree = d+1, C = 1).fit(data[0], data[1])
        poly_pred = poly.predict(data[2])
        poly_accuracy = accuracy_score(data[3], poly_pred)
        error = error_rate(poly_accuracy)
        df.loc[i,d] = error 
        df.to_csv('error_data.csv', index=False, header=False, mode='w')
        print("poly_error ",error , "degree ", i+1)

    #rbf kernel 
    c_gamma = [0.1, 1, 10, 100]
    pairs = permutations(c_gamma,2)
    counter = 3
    for c_g in pairs:
        rbf = SVC(kernel='rbf', gamma=c_g[0], C=c_g[1]).fit(data[0], data[1])
        rbf_pred = rbf.predict(data[2])
        rbf_accuracy = accuracy_score(data[3], rbf_pred)
        error = error_rate(rbf_accuracy)
        df.loc[i,counter ] = error 
        df.to_csv('error_data.csv', index=False, header=False, mode='w')
        counter += 1
        print("rbf_error ",error , "c and gamma value of  ", c_g[1], c_g[0])
    
    counter_2 = 15
    # for c = gamma vlaues -> (1 1) (.1 .1) (10 10) (100 100)
    for j in range (4):
        rbf = SVC(kernel='rbf', gamma=c_gamma[j], C=c_gamma[j]).fit(data[0], data[1])
        rbf_pred = rbf.predict(data[2])
        rbf_accuracy = accuracy_score(data[3], rbf_pred)
        error = error_rate(rbf_accuracy)
        df.loc[i,counter_2] = error 
        df.to_csv('error_data.csv', index=False, header=False, mode='w')
        counter_2 += 1
        print("rbf_error ",error , "c and gamma value of  ", c_gamma[j], c_gamma[j])

 