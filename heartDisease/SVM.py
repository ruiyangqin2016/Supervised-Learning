import csv
import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
import pydotplus
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import svm

##################################################################################################
#wine quality data set
data = pd.read_csv('heart.csv')
X = data.iloc[:, :13]
Y = data.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
features = list(X_train.columns.values)
##################################################################################################

def svm_alg():
	#SVM learning curve with RBF kernel
	# for kernel in ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed', 'callable'):
	list1=[]
	list2=[]
	list3=[]
	list4=[]
	list5=[]
	list6=[]
	list7=[]
	list8=[]
	for i in range(1,95):
	    clf = svm.SVC(kernel='poly', degree=2, gamma='scale')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list1.append(accuracy_score(y_train, clf.predict(X_train)))
	    list2.append(accuracy_score(y_test, clf.predict(X_test)))
	    clf = svm.SVC(kernel='sigmoid', gamma = 'scale')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list3.append(accuracy_score(y_train, clf.predict(X_train)))
	    list4.append(accuracy_score(y_test, clf.predict(X_test)))
	    clf = svm.SVC(kernel='rbf', gamma = 'scale')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list5.append(accuracy_score(y_train, clf.predict(X_train)))
	    list6.append(accuracy_score(y_test, clf.predict(X_test)))
	    clf = svm.SVC(kernel='linear', gamma = 'scale')
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=1 - i / 100)
	    clf.fit(X_train, y_train)
	    list7.append(accuracy_score(y_train, clf.predict(X_train)))
	    list8.append(accuracy_score(y_test, clf.predict(X_test)))
	    print('finish ', i, '...')


	plt.ylim(bottom=0,top=1.1)
	plt.plot(range(len(list1)),list1, '-r', label="Training Set('poly,2')")
	plt.plot(range(len(list2)),list2, '-r', label="Testing Set('poly,2')")
	plt.plot(range(len(list3)),list3, '-b', label="Training Set('sigmoid')")
	plt.plot(range(len(list4)),list4, '-b', label="Testing Set('sigmoid')")
	plt.plot(range(len(list5)),list5, '-y', label="Training Set('rbf')")
	plt.plot(range(len(list6)),list6, '-y', label="Testing Set('rbf')")
	plt.plot(range(len(list7)),list7, '-g', label="Training Set('linear')")
	plt.plot(range(len(list8)),list8, '-g', label="Testing Set('linear')")
	plt.legend(loc="best")
	plt.title('SVM with kernel')
	plt.ylabel('Accuracy Rate')
	plt.xlabel('Train data size')
	plt.show()
# svm_alg()

def cv_different_degree():
	list1 = []
	list2 = []
	for i in range(9):
		clf = svm.SVC(kernel='poly', degree=2, gamma='scale')
		clf = clf.fit(X_train, y_train)
		cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=True)
		mean_test = np.mean(cv_result['test_score'])
		list1.append(mean_test)

		clf = svm.SVC(kernel='linear')
		clf = clf.fit(X_train, y_train)
		cv_result = cross_validate(clf, X, Y, cv = 3, return_train_score=True)
		mean_test = np.mean(cv_result['test_score'])
		list1.append(mean_test)
		print('finish ', i,'...')
	plt.plot(range(len(list1)),list1, '-b', label="CV for degrees of poly")
	plt.legend(loc="best")
	plt.title('cross_validate for degree')
	plt.ylabel('mean test score')
	plt.xlabel('degree')
	plt.show()
# cv_different_degree()

def learning_result():
	#Choose RBF as the preferred kernel function
	clf = svm.SVC(kernel="poly", gamma='scale', degree = 2)
	clf = clf.fit(X_train, y_train)
	test_predict = clf.predict(X_test)
	print("The prediction accuracy of SVM with RBF kernel is " + str(accuracy_score(y_test, test_predict)))
	clf = svm.SVC(kernel="sigmoid", gamma='scale')
	clf = clf.fit(X_train, y_train)
	test_predict = clf.predict(X_test)
	print("The prediction accuracy of SVM with RBF kernel is " + str(accuracy_score(y_test, test_predict)))
	clf = svm.SVC(kernel="rbf", gamma='scale')
	clf = clf.fit(X_train, y_train)
	test_predict = clf.predict(X_test)
	print("The prediction accuracy of SVM with RBF kernel is " + str(accuracy_score(y_test, test_predict)))
	clf = svm.SVC(kernel="linear", gamma='scale')
	clf = clf.fit(X_train, y_train)
	test_predict = clf.predict(X_test)
	print("The prediction accuracy of SVM with RBF kernel is " + str(accuracy_score(y_test, test_predict)))
learning_result()


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

# clf = svm.SVC(kernel='poly', degree=2, gamma='scale')
# plot_learning_curve(clf, "learning curve with kerenal is poly", X, Y, ylim=[0, 1])

# clf = svm.SVC(kernel='linear')
# plot_learning_curve(clf, "learning curve with kerenal is linear", X, Y, ylim=[0, 1])

# clf = svm.SVC(kernel='poly', degree=2, gamma='scale')
# plot_learning_curve(clf, "learning curve with kerenal is poly = 2", X, Y, ylim=[0, 1])