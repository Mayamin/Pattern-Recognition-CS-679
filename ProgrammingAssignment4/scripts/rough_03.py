from libsvm.svmutil import *
from pathlib import Path
import pandas as pd
import numpy as np
import csv
def svm_fold_x(tr_path,Ttr_path,ts_path,Tts_path,img_no,fold_no):

    data = []

    if (fold_no==1):
        dataframe1 = pd.read_csv(tr_path)
        dataframe1.to_csv('trPCA_01.csv', index = None)
        tr_01 = [[] for i in range(img_no)]
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
        # print("X_train.shape ",X_train.shape )

        #read train img_label
        dataframe2 = pd.read_csv(Ttr_path)
        dataframe2.to_csv('TtrPCA_01.csv', index = None)

        with open('TtrPCA_01.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            first_row = next(csv_reader)
            y_train = np.array(first_row[0].split())
        # print("train_labels",y_train) # print("train_labels[0].type",type(y_train[0])) ////str
        # print("train_labels.shape",y_train.shape)

        #read test img coeff
        dataframe3 = pd.read_csv(ts_path)
        dataframe3.to_csv('tsPCA_01.csv', index = None)
        ts_01 = [[] for i in range(img_no)]
        X_test = []

        with open('tsPCA_01.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                row_arr = row[0].split()
                row_arr = np.array([ float(i) for i in row_arr ])
                for i in range( len(row_arr)):
                    ts_01[i].append(row_arr[i])
            X_test = np.transpose(np.array(ts_01))[:, :30]
            # print("X_test", X_test)
        # print("X_test.shape ",X_test.shape )
        
        #read test img_label
        dataframe4 = pd.read_csv(Tts_path)
        dataframe4.to_csv('TtsPCA_01.csv', index = None)
        with open('TtsPCA_01.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            first_row = next(csv_reader)
            y_test = np.array(first_row[0].split())
        # print(" y_test.shape, type(y_test[0]), y_test ", y_test.shape,type(y_test[0]), y_test)
        
        data = [X_train, y_train, X_test, y_test]
        return data
    

    if (fold_no==2):
        dataframe1 = pd.read_csv(tr_path)
        dataframe1.to_csv('trPCA_02.csv', index = None)
        tr_02 = [[] for i in range(img_no)]
        X_train = []

        with open('trPCA_02.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                row_arr = row[0].split()
                row_arr = np.array([ float(i) for i in row_arr ])
                for i in range( len(row_arr)):
                    tr_02[i].append(row_arr[i])         
            X_train = np.transpose(np.array(tr_02))[:, :30]
        # print("X_train.shape ",X_train.shape )

        #read train img_label
        dataframe2 = pd.read_csv(Ttr_path)
        dataframe2.to_csv('TtrPCA_02.csv', index = None)
        with open('TtrPCA_02.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            first_row = next(csv_reader)
            y_train = np.array(first_row[0].split())
        # print("train_labels.shape",y_train.shape)

        #read test img coeff
        dataframe3 = pd.read_csv(ts_path)
        dataframe3.to_csv('tsPCA_02.csv', index = None)
        ts_02 = [[] for i in range(img_no)]
        X_test = []

        with open('tsPCA_02.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                row_arr = row[0].split()
                row_arr = np.array([ float(i) for i in row_arr ])
                for i in range( len(row_arr)):
                    ts_02[i].append(row_arr[i])
            X_test = np.transpose(np.array(ts_02))[:, :30]
        # print("X_test.shape ",X_test.shape )
        
        #read test img_label
        dataframe4 = pd.read_csv(Tts_path)
        dataframe4.to_csv('TtsPCA_02.csv', index = None)
        with open('TtsPCA_02.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            first_row = next(csv_reader)
            y_test = np.array(first_row[0].split())
        # print(" y_test.shape, type(y_test[0]), y_test ", y_test.shape,type(y_test[0]), y_test)
        
        data = [X_train, y_train, X_test, y_test]
        return data
    
    if (fold_no==3):
        dataframe1 = pd.read_csv(tr_path)
        dataframe1.to_csv('trPCA_03.csv', index = None)
        tr_03 = [[] for i in range(img_no)]
        X_train = []

        with open('trPCA_03.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                row_arr = row[0].split()
                row_arr = np.array([ float(i) for i in row_arr ])
                for i in range( len(row_arr)):
                    tr_03[i].append(row_arr[i])          
            X_train = np.transpose(np.array(tr_03))[:, :30]
        # print("X_train.shape ",X_train.shape )

        #read train img_label
        dataframe2 = pd.read_csv(Ttr_path)
        dataframe2.to_csv('TtrPCA_03.csv', index = None)

        with open('TtrPCA_03.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            first_row = next(csv_reader)
            y_train = np.array(first_row[0].split())
        # print("train_labels.shape",y_train.shape)

        #read test img coeff
        dataframe3 = pd.read_csv(ts_path)
        dataframe3.to_csv('tsPCA_03.csv', index = None)
        ts_03 = [[] for i in range(img_no)]
        X_test = []

        with open('tsPCA_03.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                row_arr = row[0].split()
                row_arr = np.array([ float(i) for i in row_arr ])
                for i in range( len(row_arr)):
                    ts_03[i].append(row_arr[i])
            X_test = np.transpose(np.array(ts_03))[:, :30]
            # print("X_test", X_test)
        # print("X_test.shape ",X_test.shape )
        
        #read test img_label
        dataframe4 = pd.read_csv(Tts_path)
        dataframe4.to_csv('TtsPCA_03.csv', index = None)
        with open('TtsPCA_03.csv','r') as csv_file:
            csv_reader = csv.reader(csv_file)
            first_row = next(csv_reader)
            y_test = np.array(first_row[0].split())
        # print(" y_test.shape, type(y_test[0]), y_test ", y_test.shape,type(y_test[0]), y_test)
        
        data = [X_train, y_train, X_test, y_test]
        return data
