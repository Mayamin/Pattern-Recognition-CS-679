from libsvm.svmutil import *
from pathlib import Path
import pandas as pd
import numpy as np
import csv

def error_rate(accuracy):
    return (1-accuracy)*100

def svm_fold_x(img_no,fold_no,val_img_no,resolution):

    data = []
   
    # if (fold_no==1):
    # path_tr01 = Path("GenderDataRowOrder/16_20/trPCA_0" + str(fold_no) + ".txt")
    # path_Ttr01 = Path("GenderDataRowOrder/16_20/TtrPCA_0" + str(fold_no) + ".txt")
    # path_ts01 = Path("GenderDataRowOrder/16_20/tsPCA_0" + str(fold_no) + ".txt")
    # path_Tts01 = Path("GenderDataRowOrder/16_20/TtsPCA_0" + str(fold_no) + ".txt")
    # path_val_01 = Path("GenderDataRowOrder/16_20/valPCA_0" + str(fold_no) + ".txt")
    # path_Tval_01 = Path("GenderDataRowOrder/16_20/TvalPCA_0" + str(fold_no) + ".txt")
    path_tr01 = Path("GenderDataRowOrder/" +str(resolution)+"/trPCA_0" + str(fold_no) + ".txt")
    path_Ttr01 = Path("GenderDataRowOrder/" +str(resolution)+"/TtrPCA_0" + str(fold_no) + ".txt")
    path_ts01 = Path("GenderDataRowOrder/" +str(resolution)+"/tsPCA_0" + str(fold_no) + ".txt")
    path_Tts01 = Path("GenderDataRowOrder/" +str(resolution)+"/TtsPCA_0" + str(fold_no) + ".txt")
    path_val_01 = Path("GenderDataRowOrder/" +str(resolution)+"/valPCA_0" + str(fold_no) + ".txt")
    path_Tval_01 = Path("GenderDataRowOrder/" +str(resolution)+"/TvalPCA_0" + str(fold_no) + ".txt")
    
    #read train image coefficient
    dataframe1 = pd.read_csv(path_tr01)
    dataframe1.to_csv('trPCA_0' + str(fold_no) + '.csv', index = None)
    tr_01 = [[] for i in range(img_no)]
    X_train = []
    with open('trPCA_0' + str(fold_no) + '.csv','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            row_arr = row[0].split()
            row_arr = np.array([ float(i) for i in row_arr ])
            for i in range( len(row_arr)):
                tr_01[i].append(row_arr[i])
        
        X_train = np.transpose(np.array(tr_01))[:, :30]
    
    #read train img_label
    dataframe2 = pd.read_csv(path_Ttr01)
    dataframe2.to_csv('TtrPCA_0' + str(fold_no) + '.csv', index = None)
    with open('TtrPCA_0' + str(fold_no) + '.csv','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        first_row = next(csv_reader)
        y_train = np.array(first_row[0].split())
    # print("train_labels",y_train) # print("train_labels[0].type",type(y_train[0])) ////str
    # print("train_labels.shape",y_train.shape)

    
    #read validation image coefficient
    dataframe_val_1 = pd.read_csv(path_val_01)
    dataframe_val_1.to_csv('valPCA_0' + str(fold_no) + '.csv', index = None)
    val_01 = [[] for i in range(133)]

    with open('valPCA_0' + str(fold_no) + '.csv','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        iter = 0
        for row in csv_reader:
            row_arr = row[0].split()
            row_arr = np.array([ float(i) for i in row_arr ])
            for i in range( len(row_arr)):
                val_01[iter].append(row_arr[i])
            iter += 1
        val_01 = np.array([ np.array(i) for i in val_01 ])

        # print(np.atleast_2d(val_01).shape)
        X_val = np.atleast_2d(val_01)[:, :30]
        # print(iter)
    
    #read val img_label
    dataframe_val_2 = pd.read_csv(path_Tval_01)
    dataframe_val_2.to_csv('TvalPCA_0' + str(fold_no) + '.csv', index = None)
    with open('TvalPCA_0' + str(fold_no) + '.csv','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        first_row = next(csv_reader)
        y_val = np.array(first_row[0].split())

    #read test img coeff
    dataframe3 = pd.read_csv(path_ts01)
    dataframe3.to_csv('tsPCA_0' + str(fold_no) + '.csv', index = None)
    ts_01 = [[] for i in range(133)]
    X_test = []

    with open('tsPCA_0' + str(fold_no) + '.csv','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        iter = 0
        for row in csv_reader:
            row_arr = row[0].split()
            row_arr = np.array([ float(i) for i in row_arr ])
            for i in range( len(row_arr)):
                ts_01[iter].append(row_arr[i])

            iter += 1

        ts_01 = np.array([ np.array(i) for i in ts_01 ])

        # print(np.atleast_2d(ts_01).shape)
        X_test = np.atleast_2d(ts_01)[:, :30]
        # print("X_test", X_test)
    # print("X_test.shape ",X_test.shape )
    
    #read test img_label
    dataframe4 = pd.read_csv(path_Tts01)
    dataframe4.to_csv('TtsPCA_0' + str(fold_no) + '.csv', index = None)
    with open('TtsPCA_0' + str(fold_no) + '.csv','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        first_row = next(csv_reader)
        y_test = np.array(first_row[0].split())
    # print(" y_test.shape, type(y_test[0]), y_test ", y_test.shape,type(y_test[0]), y_test)
    
    data = [X_train, y_train, X_test, y_test, X_val, y_val]
    # print("X_train ",X_train, "\nfold_no ", fold_no)
    # print("y_train ",y_train)
    # print("y_val ",y_val)
    # print("y_test ",y_test)
    return data
    
