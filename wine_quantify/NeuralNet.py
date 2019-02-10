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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

##################################################################################################
#wine quality data set
data = pd.read_csv('winequality-white.csv')
X = data.iloc[:, [0,1,2,5,9,10]]
Y = data.iloc[:, 12]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
features = list(x_train.columns.values)
##################################################################################################

def crossvalidation_hiddenLayerSizes():
	scoreList = []
	layers = (12)
	clf = MLPClassifier(random_state=0, hidden_layer_sizes = layers)
	clf = clf.fit(x_train, y_train)
	cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
	mean = np.mean(cv_result['test_score'])
	print('hidden_layer: ', layers)
	scoreList.append(mean)

	layers = (12,9)
	clf = MLPClassifier(random_state=0, hidden_layer_sizes = layers)
	clf = clf.fit(x_train, y_train)
	cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
	mean = np.mean(cv_result['test_score'])
	print('hidden_layer: ', layers)
	scoreList.append(mean)

	layers = (12,9,6)
	clf = MLPClassifier(random_state=0, hidden_layer_sizes = layers)
	clf = clf.fit(x_train, y_train)
	cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
	mean = np.mean(cv_result['test_score'])
	print('hidden_layer: ', layers)
	scoreList.append(mean)

	layers = (12, 10, 8, 5)
	clf = MLPClassifier(random_state=0, hidden_layer_sizes = layers)
	clf = clf.fit(x_train, y_train)
	cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
	mean = np.mean(cv_result['test_score'])
	print('hidden_layer: ', layers)
	scoreList.append(mean)

	layers = (12, 10, 7, 5, 3)
	clf = MLPClassifier(random_state=0, hidden_layer_sizes = layers)
	clf = clf.fit(x_train, y_train)
	cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
	mean = np.mean(cv_result['test_score'])
	print('hidden_layer: ', layers)
	scoreList.append(mean)

	layers = (12, 10, 8, 6, 5, 3)
	clf = MLPClassifier(random_state=0, hidden_layer_sizes = layers)
	clf = clf.fit(x_train, y_train)
	cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
	mean = np.mean(cv_result['test_score'])
	print('hidden_layer: ', layers)
	scoreList.append(mean)

	layers = (12, 10, 8, 6, 5, 4, 3)
	clf = MLPClassifier(random_state=0, hidden_layer_sizes = layers)
	clf = clf.fit(x_train, y_train)
	cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
	mean = np.mean(cv_result['test_score'])
	print('hidden_layer: ', layers)
	scoreList.append(mean)

	layers = (12, 11, 9, 6, 5, 4, 3, 2)
	clf = MLPClassifier(random_state=0, hidden_layer_sizes = layers)
	clf = clf.fit(x_train, y_train)
	cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
	mean = np.mean(cv_result['test_score'])
	print('hidden_layer: ', layers)
	scoreList.append(mean)
	

	plt.plot(range(len(scoreList)), scoreList, '-r', label='cross_validation')
	plt.legend(loc='best')
	plt.title('cross validation of hidden_layers')
	plt.ylabel('mean test score')
	plt.xlabel('number of hidden layers')
	plt.show()
# crossvalidation_hiddenLayerSizes()

def crossvalidation_activation():
	scoreList = []
	for activation in {'identity', 'logistic', 'tanh', 'relu'}:
		clf = MLPClassifier(random_state=0, hidden_layer_sizes = (12, 10, 7, 5, 3), activation = activation)
		clf = clf.fit(x_train, y_train)
		cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=False)
		mean = np.mean(cv_result['test_score'])
		print('finish ', activation, 'mean is ', mean)
		scoreList.append(mean)
	plt.plot(range(len(scoreList)), scoreList, '-r', label='cross_validation')
	plt.legend(loc='best')

	plt.title('cross validation of activation function')
	plt.ylabel('mean test score')
	plt.xlabel('activation')
	plt.show()
# crossvalidation_activation()

def neuralnet_layers():
	list1=[]
	list2=[]
	list3=[]
	list4=[]
	list5=[]
	list6=[]
	list7=[]
	list8=[]
	for i in range(1,95):
	    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10, 7, 5, 3), random_state=1, activation='tanh')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list1.append(accuracy_score(y_train, clf.predict(X_train)))
	    list2.append(accuracy_score(y_test, clf.predict(X_test)))
	    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10, 5), random_state=1, activation='tanh')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list3.append(accuracy_score(y_train, clf.predict(X_train)))
	    list4.append(accuracy_score(y_test, clf.predict(X_test)))
	    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 4), random_state=1, activation='tanh')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list5.append(accuracy_score(y_train, clf.predict(X_train)))
	    list6.append(accuracy_score(y_test, clf.predict(X_test)))
	    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10), random_state=1, activation='tanh')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list7.append(accuracy_score(y_train, clf.predict(X_train)))
	    list8.append(accuracy_score(y_test, clf.predict(X_test)))

	plt.plot(range(len(list1)),list1, '-b', label="Training Set(12,10,7,5,3)")
	plt.plot(range(len(list2)),list2, '-b', label="Testing Set(12,10,7,5,3)")
	plt.plot(range(len(list3)),list3, '-r', label="Training Set(12,10,5)")
	plt.plot(range(len(list4)),list4, '-r', label="Testing Set(12,10,5)")
	plt.plot(range(len(list5)),list5, '-y', label="Training Set(12,4)")
	plt.plot(range(len(list6)),list6, '-y', label="Testing Set(12,4)")
	plt.plot(range(len(list7)),list7, '-g', label="Training Set(10)")
	plt.plot(range(len(list8)),list8, '-g', label="Testing Set(10)")
	plt.legend(loc="best")
	plt.title('Neural Net with different hidden layers')
	plt.ylabel('Accuracy Rate')
	plt.xlabel('Train data size')
	plt.show()
neuralnet_layers()

def neuralnet_activation():
	list1=[]
	list2=[]
	list3=[]
	list4=[]
	list5=[]
	list6=[]
	list7=[]
	list8=[]
	for i in range(1,95):
	    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10, 7, 5, 3), random_state=1, activation='relu')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list1.append(accuracy_score(y_train, clf.predict(X_train)))
	    list2.append(accuracy_score(y_test, clf.predict(X_test)))
	    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10, 7, 5, 3), random_state=1, activation='identity')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list3.append(accuracy_score(y_train, clf.predict(X_train)))
	    list4.append(accuracy_score(y_test, clf.predict(X_test)))
	    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10, 7, 5, 3), random_state=1, activation='logistic')
	    # X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    # clf.fit(X_train, y_train)
	    # list5.append(accuracy_score(y_train, clf.predict(X_train)))
	    # list6.append(accuracy_score(y_test, clf.predict(X_test)))
	    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10, 7, 5, 3), random_state=1, activation='tanh')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list7.append(accuracy_score(y_train, clf.predict(X_train)))
	    list8.append(accuracy_score(y_test, clf.predict(X_test)))

	plt.plot(range(len(list1)),list1, '-b', label="Training Set(relu)")
	plt.plot(range(len(list2)),list2, '-b', label="Testing Set(relu)")
	plt.plot(range(len(list3)),list3, '-r', label="Training Set(identity)")
	plt.plot(range(len(list4)),list4, '-r', label="Testing Set(identity)")
	# plt.plot(range(len(list5)),list5, '-y', label="Training Set(logistic)")
	# plt.plot(range(len(list6)),list6, '-y', label="Testing Set(logistic)")
	plt.plot(range(len(list7)),list7, '-g', label="Training Set(tanh)")
	plt.plot(range(len(list8)),list8, '-g', label="Testing Set(tanh)")
	plt.legend(loc="best")
	plt.title('Neural Net with 5 hidden layers(12, 10, 7, 5, 3)')
	plt.ylabel('Accuracy Rate')
	plt.xlabel('Train data size')
	plt.show()
# neuralnet_activation()

def result():
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10, 7, 5, 3), random_state=1, activation='relu')	
	clf.fit(x_train, y_train)
	print("The prediction accuracy of neural net(relu) is " + str(accuracy_score(y_test, clf.predict(x_test))))

	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10, 7, 5, 3), random_state=1, activation='identity')	
	clf.fit(x_train, y_train)
	print("The prediction accuracy of neural net(identity) is " + str(accuracy_score(y_test, clf.predict(x_test))))

	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10, 7, 5, 3), random_state=1, activation='tanh')	
	clf.fit(x_train, y_train)
	print("The prediction accuracy of neural net(tanh) is " + str(accuracy_score(y_test, clf.predict(x_test))))
# result()

# print('Please uncomment some functions to run!')