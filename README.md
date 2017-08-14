# machine-learning-algoritms-for-gender-classification
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data(hiegt,weight and shoe size) and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Test data

X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
Y_test=['male','male','male','female','female','female','male','male']

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)

# Testing using the same data
pred_tree = clf_tree.predict(X_test)
acc_tree = accuracy_score(Y_test,pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(Y_test, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(X_test)
acc_per = accuracy_score(Y_test, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(X_test)
acc_KNN = accuracy_score(Y_test, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))
