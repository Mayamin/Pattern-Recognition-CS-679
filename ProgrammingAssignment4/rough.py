# from sklearn import svm, datasets
# import sklearn.model_selection as model_selection
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# import numpy as np

# iris = datasets.load_iris()

# X = iris.data[:, :2]
# y = iris.target
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)

# print("X_train.shape",X_train.shape)
# print("y_train.shape",y_train.shape)
# print("X_test.shape",X_test.shape)
# print("y_test.shape",y_test.shape)

# print("X_train type",type(X_train))

# rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
# poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)

# poly_pred = poly.predict(X_test)
# rbf_pred = rbf.predict(X_test)

# poly_accuracy = accuracy_score(y_test, poly_pred)
# poly_f1 = f1_score(y_test, poly_pred, average='weighted')
# print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
# print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

def abc():
    print("Hello")
