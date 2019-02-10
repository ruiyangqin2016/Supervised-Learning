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
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

data = pd.read_csv('winequality-white.csv')
# X = data.iloc[:, :11]
X = data.iloc[:, [0,1,2,5,9,10]]
Y = data.iloc[:, 12]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

"""
	processing Y, if quality > 5, 1
					 quality <=5, 0

# Y_process = []
# for i in range(len(Y.values)):
# 	if Y.values[i] >5:
# 		Y_process.append(1)
# 	else:
# 		Y_process.append(0)
# # print(Y_process)
# with open('temp.csv', "w") as output:
# 	writer = csv.writer(output, lineterminator='\n')
# 	for val in Y_process:
# 		writer.writerow([val])
# print('finished')
"""

# features = list(X.columns.values)
# X = preprocessing.scale(X)
# pd.DataFrame(X, columns = features).to_csv('temp.csv', index = False)
# data = pd.read_csv('temp.csv')
# X = data.iloc[:,:6]
# print(X)
# print(X_train)
# X_train_scaled = preprocessing.scale(X_train)
# X_test_scaled = preprocessing.scale(X_test)
# print(x_train)
# x_scaler = preprocessing.StandardScaler().fit(x_train)
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train) 
# X_test_scaled = scaler.transform(X_test)


def result():
	clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=30)#, max_leaf_nodes = 5)	
	clf.fit(X_train, y_train)
	print("The prediction accuracy of neural net is " + str(accuracy_score(y_test, clf.predict(X_test))))
# result()

def crossvalidation():
	counter = 1
	scoreList = []
	list1 = []
	list2 = []
	while counter < 50:
		# crossvalidation for maxDepth
		clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=counter)
		clf = clf.fit(X_train, y_train)
		cv_result = cross_validate(clf, X, Y, cv = 5, return_train_score=True)
		mean_test = np.mean(cv_result['test_score'])
		list2.append(mean_test)
		counter = counter + 1

		# corssvalidation for max_leaf_nodes
		clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_leaf_nodes = counter)
		clf = clf.fit(X_train, y_train)
		cv_result = cross_validate(clf, X, Y, cv = 5, return_train_score=True)
		mean_test = np.mean(cv_result['test_score'])
		list1.append(mean_test)
	plt.plot(range(len(list2)),list2, '-b', label="CV for max_depth")
	plt.plot(range(len(list1)),list1, '-r', label="CV for max_leaf_nodes")
	plt.legend(loc="best")
	plt.title('cross validation')
	plt.ylabel('mean test score')
	plt.xlabel('counter')
	plt.show()
# crossvalidation()

def wineQuality():
	list1=[]
	list2=[]
	list3=[]
	list4=[]
	list5=[]
	list6=[]
	list7=[]
	list8=[]
	for i in range(1,95):
	    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5)#, max_leaf_nodes = 5)
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size = 1 - i / 100)
	    clf = clf.fit(X_train, y_train) #After being fitted, the model can then be used to predict the class of samples
	    list1.append(accuracy_score(y_train, clf.predict(X_train)))
	    list2.append(accuracy_score(y_test, clf.predict(X_test)))

	    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=10)#, max_leaf_nodes = 5)
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size = 1 - i / 100)
	    clf = clf.fit(X_train, y_train) #After being fitted, the model can then be used to predict the class of samples
	    list3.append(accuracy_score(y_train, clf.predict(X_train)))
	    list4.append(accuracy_score(y_test, clf.predict(X_test)))

	    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=15)#, max_leaf_nodes = 5)
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size = 1 - i / 100)
	    clf = clf.fit(X_train, y_train) #After being fitted, the model can then be used to predict the class of samples
	    list5.append(accuracy_score(y_train, clf.predict(X_train)))
	    list6.append(accuracy_score(y_test, clf.predict(X_test)))

	    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=3)#, max_leaf_nodes = 5)
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size = 1 - i / 100)
	    clf = clf.fit(X_train, y_train) #After being fitted, the model can then be used to predict the class of samples
	    list7.append(accuracy_score(y_train, clf.predict(X_train)))
	    list8.append(accuracy_score(y_test, clf.predict(X_test)))
	plt.plot(range(len(list1)),list1, '-b', label="Training Set maxDepth = 5")
	plt.plot(range(len(list2)),list2, '-b', label="Testing Set, maxDepth = 5")
	plt.plot(range(len(list3)),list3, '-r', label="Training Set maxDepth = 10")
	plt.plot(range(len(list4)),list4, '-r', label="Testing Set, maxDepth = 10")
	plt.plot(range(len(list5)),list5, '-y', label="Training Set maxDepth = 15")
	plt.plot(range(len(list6)),list6, '-y', label="Testing Set, maxDepth = 15")
	plt.plot(range(len(list7)),list7, '-g', label="Training Set maxDepth = 3")
	plt.plot(range(len(list8)),list8, '-g', label="Testing Set, maxDepth = 3")
	plt.legend(loc="best")
	plt.title('Decision Tree with different max_depth')
	plt.ylabel('Accuracy Rate')
	plt.xlabel('Train data size')
	plt.show()
wineQuality()

def visual():
	iris = load_iris()
	clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5, max_leaf_nodes = 5)
	clf = clf.fit(iris.data, iris.target)
	dot_data = tree.export_graphviz(clf, out_file=None) 
	graph = graphviz.Source(dot_data) 
	graph.render("iris") 

	dot_data = tree.export_graphviz(clf, out_file=None, 
	                     feature_names=iris.feature_names,  
	                     class_names=iris.target_names,
	                     filled=True, rounded=True,  
	                     special_characters=True)  
	print('pdf file output!')
	graph = graphviz.Source(dot_data)  
	graph
	
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

# learning curve
# clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5, max_leaf_nodes = 5)
# plot_learning_curve(clf, "Decision Tree with max depth 5", X, Y, ylim=[0, 1])

# visual()