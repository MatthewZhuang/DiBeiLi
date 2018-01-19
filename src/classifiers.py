#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
 @description:
        
 @Time       : 17/7/4 上午12:26
 @Author     : guomianzhuang
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import preprocess as pre
from sklearn import metrics


h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

data = pre.load_data()
data = pre.process_without_discretize(data)
X_train, X_test, y_train, y_test = pre.generate_corpus_for_continues(data)

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print score
    labels = clf.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(y_test, labels)))


"""
/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 /Users/Matthew/Documents/workspace/project/DiBeiLi/src/classifiers.py

 @description:

 @Time       : 17/7/4 上午12:26
 @Author     : guomianzhuang

fitered:126
0.724860705757
Classification report for classifier KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='uniform'):
             precision    recall  f1-score   support

        0.0       0.69      0.64      0.66      1594
        1.0       0.75      0.79      0.77      2175

avg / total       0.72      0.72      0.72      3769


0.603077739453
Classification report for classifier SVC(C=0.025, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False):
             precision    recall  f1-score   support

        0.0       0.57      0.26      0.36      1594
        1.0       0.61      0.85      0.71      2175

avg / total       0.59      0.60      0.56      3769


0.675776067923
Classification report for classifier SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=2, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False):
             precision    recall  f1-score   support

        0.0       0.74      0.36      0.48      1594
        1.0       0.66      0.91      0.76      2175

avg / total       0.69      0.68      0.64      3769


0.736269567525
Classification report for classifier DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'):
             precision    recall  f1-score   support

        0.0       0.70      0.66      0.68      1594
        1.0       0.76      0.79      0.78      2175

avg / total       0.73      0.74      0.74      3769


0.633589811621
Classification report for classifier RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=5, max_features=1, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False):
             precision    recall  f1-score   support

        0.0       0.78      0.19      0.30      1594
        1.0       0.62      0.96      0.75      2175

avg / total       0.68      0.63      0.56      3769


0.706553462457
Classification report for classifier MLPClassifier(activation='relu', alpha=1, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False):
             precision    recall  f1-score   support

        0.0       0.68      0.58      0.63      1594
        1.0       0.72      0.80      0.76      2175

avg / total       0.70      0.71      0.70      3769


0.729901830724
Classification report for classifier AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None):
             precision    recall  f1-score   support

        0.0       0.69      0.64      0.67      1594
        1.0       0.75      0.79      0.77      2175

avg / total       0.73      0.73      0.73      3769


0.535420535951
Classification report for classifier GaussianNB(priors=None):
             precision    recall  f1-score   support

        0.0       0.47      0.71      0.56      1594
        1.0       0.66      0.41      0.50      2175

avg / total       0.58      0.54      0.53      3769


/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sklearn/discriminant_analysis.py:694: UserWarning: Variables are collinear
  warnings.warn("Variables are collinear")
0.647386574688
Classification report for classifier QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
               store_covariances=False, tol=0.0001):
             precision    recall  f1-score   support

        0.0       0.63      0.41      0.50      1594
        1.0       0.66      0.82      0.73      2175

avg / total       0.64      0.65      0.63      3769



Process finished with exit code 0

"""