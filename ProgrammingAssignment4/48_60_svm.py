
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline
import csv
from svm_fold import *
from itertools import permutations

total_fold_number = 3
poly_degrees = 3
C_parameters = [.1,1,10,100]
D_parameters = [1,2,3]
svm_p1_opti = []
svm_p2_opti = []
svm_p3_opti = []
svm_R1_opti = []
svm_R2_opti = []
svm_R3_opti = []

resolution_1 = "48_60"


error_poly_rbf_f123 = [
    [0] * 28,
    [0] * 28,
    [0] * 28
]

#for validation
with open('error_val_data_48_60.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in error_poly_rbf_f123:
        writer.writerow(row)
df = pd.read_csv('error_val_data_48_60.csv', header=None)
flag_poly = 0


for i in range(total_fold_number):
    img_no = 134
    val_img_no = 133
    fold_no = i+1
    data= svm_fold_x(img_no,fold_no,val_img_no,resolution_1)

    max_accuracy = -1
    svm_best_poly = None

    # print(" Fold no ", i + 1)
    for d in D_parameters:
        for c in C_parameters:
            # print(" ploy d = ", d, "poly C ", c)
            poly = make_pipeline(MaxAbsScaler(), SVC(kernel='poly', degree = d, C = c)).fit(data[0], data[1])
            poly_pred = poly.predict(data[4])
            poly_accuracy = accuracy_score(data[5], poly_pred)

            if (poly_accuracy > max_accuracy):
                svm_best_poly = poly
                max_accuracy = poly_accuracy

            error = error_rate(poly_accuracy)
            df.loc[i,flag_poly] = error 
            df.to_csv('error_val_data_48_60.csv', index=False, header=False, mode='w')
            flag_poly += 1
    flag_poly = 0
        
    if (i == 0):
        svm_p1_opti.append(svm_best_poly)
    elif (i == 1):
        svm_p2_opti.append(svm_best_poly)
    else:
        svm_p3_opti.append(svm_best_poly)

    #rbf kernel 
    c_gamma = [0.1, 1, 10, 100]
    pairs = permutations(c_gamma,2)
    # print(pairs)

    max_accuracy = -1
    svm_best_rbf = None

    counter = 12
    for c_g in pairs:
        rbf = make_pipeline(MaxAbsScaler(), SVC(kernel='rbf', gamma=c_g[0], C=c_g[1])).fit(data[0], data[1])
        # print("gamma ", c_g[0] , " C ", c_g[1])

        rbf_pred = rbf.predict(data[4])
        rbf_accuracy = accuracy_score(data[5], rbf_pred)

        if (rbf_accuracy > max_accuracy):
            svm_best_rbf = rbf
            max_accuracy = rbf_accuracy

        error = error_rate(rbf_accuracy)

        # print(f':) {rbf_accuracy} + {error / 100} = {rbf_accuracy + (error / 100) }')
        df.loc[i,counter] = error 
        df.to_csv('error_val_data_48_60.csv', index=False, header=False, mode='w')
        counter += 1
        
    
    if (i == 0):
        svm_R1_opti.append(svm_best_rbf)
    elif (i == 1):
        svm_R2_opti.append(svm_best_rbf)
    else:
        svm_R3_opti.append(svm_best_rbf)

    # counter = 12
    counter_2 = 24
    # for c = gamma vlaues -> (1 1) (.1 .1) (10 10) (100 100)
    for j in range (4):
        # print("gamma ", c_gamma[j], " C ", c_gamma[j])
        rbf = SVC(kernel='rbf', gamma=c_gamma[j], C=c_gamma[j]).fit(data[0], data[1])
        rbf_pred = rbf.predict(data[4])
        rbf_accuracy = accuracy_score(data[5], rbf_pred)
        error = error_rate(rbf_accuracy)
        df.loc[i,counter_2] = error 
        df.to_csv('error_val_data_48_60.csv', index=False, header=False, mode='w')
        counter_2 += 1
        
    # counter_2 = 0
fold_col_error = [
    [ [], [] ] ,
    [ [], [] ] ,
    [ [], [] ]
]
row = 0 
col = 0

#reading min data from val_error file
with open('error_val_data_48_60.csv', 'r') as file:
    reader = csv.reader(file)
    for i,row in enumerate(reader):
        float_row = [ float(j) for j in row ]
        min_val_poly = min(float_row[:11])
        # print(float_row[:11])
        min_val_rbf = min(float_row[12:])
        # print(float_row[12:])
        mindex_poly = float_row.index(min_val_poly)
        mindex_rbf = float_row.index(min_val_rbf)
        fold_col_error[i][0].append([i+1, mindex_poly, min_val_poly])
        fold_col_error[i][1].append([i+1, mindex_rbf, min_val_rbf])
        # print(f"Fold no {i+1}: Minimum value is {min_val_poly}, found in column {mindex_poly}")
        # print(f"Fold no {i+1}: Minimum value is {min_val_rbf}, found in column {mindex_rbf}")

#for test data
with open('error_data_48_60.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in error_poly_rbf_f123:
        writer.writerow(row)
df = pd.read_csv('error_data_48_60.csv', header=None)

for i in range(total_fold_number):
    img_no = 134
    val_img_no = 133
    fold_no = i+1
    data= svm_fold_x(img_no,fold_no,val_img_no,resolution_1)

    
    if i == 0 : #for fold 1
        #polynomial kernel 
        poly_pred = svm_p1_opti[0].predict(data[2])
        poly_accuracy = accuracy_score(data[3], poly_pred)
        error = error_rate(poly_accuracy)
        df.loc[i,0] = error 
        df.to_csv('error_data_48_60.csv', index=False, header=False, mode='w')
        
        #rbf kernel 
        c_gamma = [0.1, 1, 10, 100]
        pairs = permutations(c_gamma,2)
        rbf_pred = svm_R1_opti[0].predict(data[2])
        rbf_accuracy = accuracy_score(data[3], rbf_pred)
        error = error_rate(rbf_accuracy)
        df.loc[i,1] = error 
        df.to_csv('error_data_48_60.csv', index=False, header=False, mode='w')
    
    if i == 1 : #for fold 2
        #polynomial kernel 
        poly_pred = svm_p2_opti[0].predict(data[2])
        poly_accuracy = accuracy_score(data[3], poly_pred)
        error = error_rate(poly_accuracy)
        df.loc[i,0] = error 
        df.to_csv('error_data_48_60.csv', index=False, header=False, mode='w')
        
        #rbf kernel 
        c_gamma = [0.1, 1, 10, 100]
        pairs = permutations(c_gamma,2)
        rbf_pred = svm_R2_opti[0].predict(data[2])
        rbf_accuracy = accuracy_score(data[3], rbf_pred)
        error = error_rate(rbf_accuracy)
        df.loc[i,1] = error 
        df.to_csv('error_data_48_60.csv', index=False, header=False, mode='w')

    if i == 2 : #for fold 3
        #polynomial kernel 
        poly_pred = svm_p3_opti[0].predict(data[2])
        poly_accuracy = accuracy_score(data[3], poly_pred)
        error = error_rate(poly_accuracy)
        df.loc[i,0] = error 
        df.to_csv('error_data_48_60.csv', index=False, header=False, mode='w')
        
        #rbf kernel 
        c_gamma = [0.1, 1, 10, 100]
        pairs = permutations(c_gamma,2)
        rbf_pred = svm_R3_opti[0].predict(data[2])
        rbf_accuracy = accuracy_score(data[3], rbf_pred)
        error = error_rate(rbf_accuracy)
        df.loc[i,1] = error 
        df.to_csv('error_data_48_60.csv', index=False, header=False, mode='w')
    


