import csv
import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

##################################################################################################
#wine quality data set
data = pd.read_csv('winequality-white.csv')
X = data.iloc[:, [0,1,2,5,9,10]]
Y = data.iloc[:, 12]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
features = list(x_train.columns.values)
##################################################################################################

def crossvalidation_nNeighbors():
	neighbors = 2
	scoreList = []
	while neighbors < 200:
		clf = KNeighborsClassifier(n_neighbors = neighbors, weights = 'distance')
		clf = clf.fit(x_train, y_train)
		cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
		mean = np.mean(cv_result['test_score'])
		print('finish ', neighbors, 'mean is ', mean)
		scoreList.append(mean)
		neighbors = neighbors + 1
	plt.plot(range(len(scoreList)), scoreList, '-r', label='cross_validation')
	plt.legend(loc='best')
	plt.title('cross validation of numbers of neighbors')
	plt.ylabel('mean test score')
	plt.xlabel('neighbors')
	plt.show()
# crossvalidation_nNeighbors()

def crossvalidation_leaf_size():
	pVal = 1
	scoreList = []
	while pVal < 15:
		clf = KNeighborsClassifier(n_neighbors = 27)
		clf = clf.fit(x_train, y_train)
		cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
		mean = np.mean(cv_result['test_score'])
		print('finish ', pVal, 'mean is ', mean)
		scoreList.append(mean)
		pVal = pVal + 1
	plt.plot(range(len(scoreList)), scoreList, '-r', label='cross_validation')
	plt.legend(loc='best')
	plt.title('cross validation of p value')
	plt.ylabel('mean test score')
	plt.xlabel('neighbors')
	plt.show()
# crossvalidation_leaf_size()
def knn_alg():
	list1=[]
	list2=[]
	for i in range(1,95):
	    clf = KNeighborsClassifier(n_neighbors = 27, p = 2, weights = 'distance')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    train_predict = clf.predict(X_train)
	    test_predict = clf.predict(X_test)
	    list1.append(accuracy_score(y_train, train_predict))
	    list2.append(accuracy_score(y_test, test_predict))
	plt.plot(range(len(list1)),list1, '-r', label="Training Set")
	plt.plot(range(len(list2)),list2, '-b', label="Testing Set")
	plt.legend(loc="best")
	plt.title('KNN with n_neighbors = 27 and weights = distance')
	plt.ylabel('Accuracy Rate')
	plt.xlabel('Train data size')
	plt.show()
knn_alg()

def result():
	clf = KNeighborsClassifier(n_neighbors = 15, p = 1)	
	x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
	clf.fit(x_train, y_train)
	print("The prediction accuracy of neural net is " + str(accuracy_score(y_test, clf.predict(x_test))))
# result()