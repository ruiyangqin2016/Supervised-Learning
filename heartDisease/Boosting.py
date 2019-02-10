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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

##################################################################################################
#wine quality data set
data = pd.read_csv('heart.csv')
X = data.iloc[:, :13]
Y = data.iloc[:, 13]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
features = list(x_train.columns.values)
##################################################################################################
# base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=2)

def crossvalidation_baseEstimator():
	scoreList = []
	depth = 1
	while depth <= 20:
		base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=depth)
		clf = AdaBoostClassifier(base_estimator = base_estimator)
		clf = clf.fit(x_train, y_train)
		cv_result = cross_validate(clf, X, Y, cv = 2, return_train_score=False)
		mean = np.mean(cv_result['test_score'])
		scoreList.append(mean)
		depth += 1
	plt.plot(range(len(scoreList)), scoreList, '-r', label='cross_validation')
	plt.legend(loc='best')
	plt.title('cross validation of max_depth of base_estimator')
	plt.ylabel('mean test score')
	plt.xlabel('max_depth of base_estimator')
	plt.show()
# crossvalidation_baseEstimator()

def crossvalidation_nEstimators():
	nE = 1
	scoreList = []
	base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=8)
	while nE <= 100:
		clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=nE, )
		clf = clf.fit(x_train, y_train)
		cv_result = cross_validate(clf, X, Y, cv = 2, return_train_score=False)
		mean = np.mean(cv_result['test_score'])
		print('finish ', nE, 'mean is ', mean)
		scoreList.append(mean)
		nE = nE + 1
	plt.plot(range(len(scoreList)), scoreList, '-r', label='cross_validation')
	plt.legend(loc='best')
	plt.title('cross validation of n_estimators given base_estimator max_depth = 8')
	plt.ylabel('mean test score')
	plt.xlabel('n_estimators')
	plt.show()
# crossvalidation_nEstimators()

def boost2():
	list1=[]
	list2=[]
	list3=[]
	list4=[]
	list5=[]
	list6=[]
	list7=[]
	list8=[]
	ne = 15
	lr = 0.5
	base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=8)
	for i in range(1,95):
		clf = AdaBoostClassifier(base_estimator, n_estimators = 10)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
		clf.fit(X_train, y_train)
		train_predict = clf.predict(X_train)
		test_predict = clf.predict(X_test)
		list1.append(accuracy_score(y_train, train_predict))
		list2.append(accuracy_score(y_test, test_predict))
		clf = AdaBoostClassifier(base_estimator, n_estimators = 25)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
		clf.fit(X_train, y_train)
		train_predict = clf.predict(X_train)
		test_predict = clf.predict(X_test)
		list3.append(accuracy_score(y_train, train_predict))
		list4.append(accuracy_score(y_test, test_predict))
		clf = AdaBoostClassifier(base_estimator, n_estimators = 40)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
		clf.fit(X_train, y_train)
		train_predict = clf.predict(X_train)
		test_predict = clf.predict(X_test)
		list5.append(accuracy_score(y_train, train_predict))
		list6.append(accuracy_score(y_test, test_predict))
		clf = AdaBoostClassifier(base_estimator, n_estimators = 55)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
		clf.fit(X_train, y_train)
		train_predict = clf.predict(X_train)
		test_predict = clf.predict(X_test)
		list7.append(accuracy_score(y_train, train_predict))
		list8.append(accuracy_score(y_test, test_predict))
	plt.plot(range(len(list1)),list1, '-r', label="Training Set(n_estimators = 10)")
	plt.plot(range(len(list2)),list2, '-r', label="Testing Set(n_estimators = 10)")
	plt.plot(range(len(list7)),list7, '-g', label="Training Set(n_estimators = 25)")
	plt.plot(range(len(list8)),list8, '-g', label="Testing Set(n_estimators = 25)")
	plt.plot(range(len(list3)),list3, '-b', label="Training Set(n_estimators = 40)")
	plt.plot(range(len(list4)),list4, '-b', label="Testing Set(n_estimators = 40)")
	plt.plot(range(len(list5)),list5, '-y', label="Training Set(n_estimators = 55)")
	plt.plot(range(len(list6)),list6, '-y', label="Testing Set(n_estimators = 55)")
	plt.legend(loc="best")
	plt.title('Boosting Desicion Tree with different n_estimators')
	plt.ylabel('Accuracy Rate')
	plt.xlabel('Train data size')
	plt.show()
# boost2()

def boost():
	list1=[]
	list2=[]
	list3=[]
	list4=[]
	list5=[]
	list6=[]
	list7=[]
	list8=[]
	ne = 15
	lr = 0.5
	for i in range(1,95):
		base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5)
		clf = AdaBoostClassifier(base_estimator = base_estimator)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
		clf.fit(X_train, y_train)
		train_predict = clf.predict(X_train)
		test_predict = clf.predict(X_test)
		list1.append(accuracy_score(y_train, train_predict))
		list2.append(accuracy_score(y_test, test_predict))
		base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=6)
		clf = AdaBoostClassifier(base_estimator = base_estimator)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
		clf.fit(X_train, y_train)
		train_predict = clf.predict(X_train)
		test_predict = clf.predict(X_test)
		list3.append(accuracy_score(y_train, train_predict))
		list4.append(accuracy_score(y_test, test_predict))
		base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=7)
		clf = AdaBoostClassifier(base_estimator = base_estimator)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
		clf.fit(X_train, y_train)
		train_predict = clf.predict(X_train)
		test_predict = clf.predict(X_test)
		list5.append(accuracy_score(y_train, train_predict))
		list6.append(accuracy_score(y_test, test_predict))
		base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=8)
		clf = AdaBoostClassifier(base_estimator = base_estimator)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
		clf.fit(X_train, y_train)
		train_predict = clf.predict(X_train)
		test_predict = clf.predict(X_test)
		list7.append(accuracy_score(y_train, train_predict))
		list8.append(accuracy_score(y_test, test_predict))
	plt.plot(range(len(list1)),list1, '-r', label="Training Set(base_estimator MD = 5)")
	plt.plot(range(len(list2)),list2, '-r', label="Testing Set(base_estimator MD = 5)")
	plt.plot(range(len(list7)),list7, '-g', label="Training Set(base_estimator MD = 6)")
	plt.plot(range(len(list8)),list8, '-g', label="Testing Set(base_estimator MD = 6)")
	plt.plot(range(len(list3)),list3, '-b', label="Training Set(base_estimator MD = 7)")
	plt.plot(range(len(list4)),list4, '-b', label="Testing Set(base_estimator MD = 7)")
	plt.plot(range(len(list5)),list5, '-y', label="Training Set(base_estimator MD = 8)")
	plt.plot(range(len(list6)),list6, '-y', label="Testing Set(base_estimator MD = 8)")
	plt.legend(loc="best")
	plt.title('Boosting Desicion Tree with different base_estimator max_depth')
	plt.ylabel('Accuracy Rate')
	plt.xlabel('Train data size')
	plt.show()
# boost()

base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=8)
clf = AdaBoostClassifier(base_estimator = base_estimator)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
clf.fit(X_train, y_train)
print("The prediction accuracy of boosting is " + str(accuracy_score(y_test, clf.predict(X_test))))

base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=9)
clf = AdaBoostClassifier(base_estimator = base_estimator)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
clf.fit(X_train, y_train)
print("The prediction accuracy of boosting is " + str(accuracy_score(y_test, clf.predict(X_test))))

base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=10)
clf = AdaBoostClassifier(base_estimator = base_estimator)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
clf.fit(X_train, y_train)
print("The prediction accuracy of boosting is " + str(accuracy_score(y_test, clf.predict(X_test))))

base_estimator = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=11)
clf = AdaBoostClassifier(base_estimator = base_estimator)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
clf.fit(X_train, y_train)
print("The prediction accuracy of boosting is " + str(accuracy_score(y_test, clf.predict(X_test))))