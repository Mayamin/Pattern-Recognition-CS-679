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
from svm_fold import *
from itertools import permutations

#fold1
path_tr01 = Path("GenderDataRowOrder/16_20/trPCA_01.txt")
path_Ttr01 = Path("GenderDataRowOrder/16_20/TtrPCA_01.txt")
path_ts01 = Path("GenderDataRowOrder/16_20/tsPCA_01.txt")
path_Tts01 = Path("GenderDataRowOrder/16_20/TtsPCA_01.txt")
img_no = 134
fold_no = 1

#polynomial kernel

data= svm_fold_x(path_tr01 ,path_Ttr01,path_ts01,path_Tts01,img_no,fold_no)
#SVM train_test on poly kernel d= 1  
poly = SVC(kernel='poly', degree=1, C=1).fit(data[0], data[1])
poly_pred = poly.predict(data[2])
poly_accuracy = accuracy_score(data[3], poly_pred)
poly_f1 = f1_score(data[3], poly_pred, average='weighted')
print('Fold 1 degree 1 Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('Fold 1 degree 1 F1(Polynomial Kernel): ', "%.2f" % (poly_f1*100))

#SVM train_test on poly kernel d= 2 
poly = SVC(kernel='poly', degree=2, C=1).fit(data[0], data[1])
poly_pred = poly.predict(data[2])
poly_accuracy = accuracy_score(data[3], poly_pred)
poly_f1 = f1_score(data[3], poly_pred, average='weighted')
print('Fold 1 degree 2 Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('Fold 1 degree 2 F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

#SVM train_test on poly kernel d= 3 
poly = SVC(kernel='poly', degree=3, C=1).fit(data[0], data[1])
poly_pred = poly.predict(data[2])
poly_accuracy = accuracy_score(data[3], poly_pred)
poly_f1 = f1_score(data[3], poly_pred, average='weighted')
print('Fold 1 degree 3 Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('Fold 1 degree 3 F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

c_gamma = [0.1, 1, 10, 100]
pairs = permutations(c_gamma,2)

#rbf kernel fold 1
for c_g in pairs:
    rbf = SVC(kernel='rbf', gamma=c_g[0], C=c_g[1]).fit(data[0], data[1])
    rbf_pred = rbf.predict(data[2])
    rbf_accuracy = accuracy_score(data[3], rbf_pred)
    rbf_f1 = f1_score(data[3], rbf_pred, average='weighted')
    print(" fold 1 RBF kernel with gamma and c value of ", c_g[0],c_g[1])
    print('fold 1  Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('fold 1  F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

# for c ang gamma vlaues 1 1
for i in range (4):
    rbf = SVC(kernel='rbf', gamma=c_gamma[i], C=c_gamma[i]).fit(data[0], data[1])
    rbf_pred = rbf.predict(data[2])
    rbf_accuracy = accuracy_score(data[3], rbf_pred)
    rbf_f1 = f1_score(data[3], rbf_pred, average='weighted')
    print(" fold 1  RBF kernel with gamma and c value of ", c_gamma[i],c_gamma[i])
    print('fold 1  Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('fold 1  F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

# for i in range (4):
# #checking if gamma default .5 works for all results were the same!
#     rbf = SVC(kernel='rbf', gamma= .5, C=c_gamma[i]).fit(data[0], data[1])
#     rbf_pred = rbf.predict(data[2])
#     rbf_accuracy = accuracy_score(data[3], rbf_pred)
#     rbf_f1 = f1_score(data[3], rbf_pred, average='weighted')
#     print(" RBF kernel with gamma and c value of ", .5,c_gamma[i])
#     print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
#     print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))


#fold 2
path_tr02 = Path("GenderDataRowOrder/16_20/trPCA_02.txt")
path_Ttr02 = Path("GenderDataRowOrder/16_20/TtrPCA_02.txt")
path_ts02 = Path("GenderDataRowOrder/16_20/tsPCA_02.txt")
path_Tts02 = Path("GenderDataRowOrder/16_20/TtsPCA_02.txt")
img_no = 134
fold_no = 2

#SVM train_test on poly kernel d= 1 
data= svm_fold_x(path_tr02 ,path_Ttr02,path_ts02,path_Tts02,img_no,fold_no)
poly = SVC(kernel='poly', degree=1, C=1).fit(data[0], data[1])
poly_pred = poly.predict(data[2])
poly_accuracy = accuracy_score(data[3], poly_pred)
poly_f1 = f1_score(data[3], poly_pred, average='weighted')
print('Fold 2 degree 1  Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('Fold 2 degree 1 F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

#SVM train_test on poly kernel d= 2
data= svm_fold_x(path_tr02 ,path_Ttr02,path_ts02,path_Tts02,img_no,fold_no)
poly = SVC(kernel='poly', degree=2, C=1).fit(data[0], data[1])
poly_pred = poly.predict(data[2])
poly_accuracy = accuracy_score(data[3], poly_pred)
poly_f1 = f1_score(data[3], poly_pred, average='weighted')
print('Fold 2 degree 2  Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('Fold 2 degree 2 F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

#SVM train_test on poly kernel d= 3
data= svm_fold_x(path_tr02 ,path_Ttr02,path_ts02,path_Tts02,img_no,fold_no)
poly = SVC(kernel='poly', degree=3, C=1).fit(data[0], data[1])
poly_pred = poly.predict(data[2])
poly_accuracy = accuracy_score(data[3], poly_pred)
poly_f1 = f1_score(data[3], poly_pred, average='weighted')
print('Fold 2 degree 3  Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('Fold 2 degree 3 F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))


#rbf kernel fold 2
for c_g in pairs:
    rbf = SVC(kernel='rbf', gamma=c_g[0], C=c_g[1]).fit(data[0], data[1])
    rbf_pred = rbf.predict(data[2])
    rbf_accuracy = accuracy_score(data[3], rbf_pred)
    rbf_f1 = f1_score(data[3], rbf_pred, average='weighted')
    print(" fold 2 RBF kernel with gamma and c value of ", c_g[0],c_g[1])
    print('fold 2 Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('fold 2 F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

# for c ang gamma vlaues same same (1,1), (.1,.1), (10,10), (100,100)
for i in range (4):
    rbf = SVC(kernel='rbf', gamma=c_gamma[i], C=c_gamma[i]).fit(data[0], data[1])
    rbf_pred = rbf.predict(data[2])
    rbf_accuracy = accuracy_score(data[3], rbf_pred)
    rbf_f1 = f1_score(data[3], rbf_pred, average='weighted')
    print(" fold 2 RBF kernel with gamma and c value of ", c_gamma[i],c_gamma[i])
    print('fold 2 ccuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('fold 2 F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))



#fold 3
path_tr03 = Path("GenderDataRowOrder/16_20/trPCA_03.txt")
path_Ttr03 = Path("GenderDataRowOrder/16_20/TtrPCA_03.txt")
path_ts03 = Path("GenderDataRowOrder/16_20/tsPCA_03.txt")
path_Tts03 = Path("GenderDataRowOrder/16_20/TtsPCA_03.txt")
img_no = 134
fold_no = 3

#SVM train_test on poly kernel d= 1 
data= svm_fold_x(path_tr03 ,path_Ttr03,path_ts03,path_Tts03,img_no,fold_no)
poly = SVC(kernel='poly', degree=1, C=1).fit(data[0], data[1])
poly_pred = poly.predict(data[2])
poly_accuracy = accuracy_score(data[3], poly_pred)
poly_f1 = f1_score(data[3], poly_pred, average='weighted')
print('Fold 3 degree 1 Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('Fold 3 degree 1 F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

#SVM train_test on poly kernel d= 2
data= svm_fold_x(path_tr03 ,path_Ttr03,path_ts03,path_Tts03,img_no,fold_no)
poly = SVC(kernel='poly', degree=2, C=1).fit(data[0], data[1])
poly_pred = poly.predict(data[2])
poly_accuracy = accuracy_score(data[3], poly_pred)
poly_f1 = f1_score(data[3], poly_pred, average='weighted')
print('Fold 3 degree 2 Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('Fold 3 degree 2 F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

#SVM train_test on poly kernel d= 3
data= svm_fold_x(path_tr03 ,path_Ttr03,path_ts03,path_Tts03,img_no,fold_no)
poly = SVC(kernel='poly', degree=3, C=1).fit(data[0], data[1])
poly_pred = poly.predict(data[2])
poly_accuracy = accuracy_score(data[3], poly_pred)
poly_f1 = f1_score(data[3], poly_pred, average='weighted')
print('Fold 3 degree 3 Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('Fold 3 degree 3 F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

#rbf kernel fold 3
for c_g in pairs:
    rbf = SVC(kernel='rbf', gamma=c_g[0], C=c_g[1]).fit(data[0], data[1])
    rbf_pred = rbf.predict(data[2])
    rbf_accuracy = accuracy_score(data[3], rbf_pred)
    rbf_f1 = f1_score(data[3], rbf_pred, average='weighted')
    print(" fold 3 RBF kernel with gamma and c value of ", c_g[0],c_g[1])
    print('fold 3 Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('fold 3 F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

# for c ang gamma vlaues same same (1,1), (.1,.1), (10,10), (100,100)
for i in range (4):
    rbf = SVC(kernel='rbf', gamma=c_gamma[i], C=c_gamma[i]).fit(data[0], data[1])
    rbf_pred = rbf.predict(data[2])
    rbf_accuracy = accuracy_score(data[3], rbf_pred)
    rbf_f1 = f1_score(data[3], rbf_pred, average='weighted')
    print(" fold 3 RBF kernel with gamma and c value of ", c_gamma[i],c_gamma[i])
    print('fold 3 ccuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('fold 3 F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
