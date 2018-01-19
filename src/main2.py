#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 7/3/2017 10:45 AM
    @desc:
        continues variable
    @author: guomianzhuang
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
import preprocess as pre
import logging
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier


logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='../data/myapp.log',
                filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def parameter_define():
    knn = KNeighborsClassifier(algorithm='auto')
    svc = SVC()  # time overhead
    decision_tree = DecisionTreeClassifier(random_state=42)
    max_depth = [3, 5, 8]
    random_forest = RandomForestClassifier()
    adaboost = AdaBoostClassifier()

def search_parameters():
    logging.info("stage: main method...")
    data = pre.load_data()
    data = pre.process_without_discretize(data)
    X_train, X_test, y_train, y_test = pre.generate_corpus_for_continues(data)
    logging.info("stage: searching begin...")
    pipe = Pipeline([
        ('classify', SVC(random_state=42))
    ])
    C = [0.01, 0.1, 1, 3, 5]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    degree = [3, 5]   # for poly kernel

    param_grid = [
        {
            'classify__C': C,
            'classify__kernel': kernel,
            'classify__degree': degree
        }
    ]

    grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
    grid.fit(X_train, y_train)
    logging.info("stage: searching end...")
    print(grid.best_params_)
    estimator = grid.best_estimator_
    labels = estimator.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (estimator, metrics.classification_report(y_test, labels)))
    joblib.dump(estimator, "../models/SVC.m")


# do more analysis. the best one for now. no over fitting
def search_parameters2():
    logging.info("stage: main method...")
    # train data
    data = pre.load_data(train=True)
    data = pre.process_without_discretize(data)
    X_train, X, y_train, y = pre.generate_corpus_for_continues(data, test_size=0.0001)
    # test data
    data = pre.load_data(train=False)
    data = pre.process_without_discretize(data)
    X, X_test, y, y_test = pre.generate_corpus_for_continues(data, test_size=0.9999)


    logging.info("stage: searching begin...")
    pipe = Pipeline([
        ('classify', ExtraTreesClassifier(random_state=0))
    ])

    n_estimators = [180, 200, 250, 300, 350, 400]
    max_depth = [20, 25, 30, 40]

    param_grid = [
        {
            'classify__n_estimators': n_estimators,
            'classify__max_depth': max_depth
        }
    ]

    grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
    grid.fit(X_train, y_train)
    logging.info("stage: searching end...")
    print(grid.best_params_)
    estimator = grid.best_estimator_
    labels = estimator.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (estimator, metrics.classification_report(y_test, labels)))
    joblib.dump(estimator, "../models/ExtraTreesClassifier.m")


def train_use_searched_parameter():
    # train data
    data = pre.load_data(train=True)
    data = pre.process_without_discretize(data)
    X_train, X, y_train, y = pre.generate_corpus_for_continues(data, test_size=0.0001)
    # test data
    data = pre.load_data(train=False)
    data = pre.process_without_discretize(data)
    X, X_test, y, y_test = pre.generate_corpus_for_continues(data, test_size=0.9999)

    logging.info("stage: searching begin...")
    model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=15)
    model.fit(X_train, y_train)
    print(model.feature_importances_)
    labels = model.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (model, metrics.classification_report(y_test, labels)))
    # joblib.dump(model, "../models/RandomForestClassifier.m")


def load_model():
    data = pre.load_data(train=False)
    data = pre.process_without_discretize(data)
    X, X_test, y, y_test = pre.generate_corpus_for_continues(data, test_size=0.9999)
    model = joblib.load("../models/MLPClassifier.m")
    # model.fit(X_test, y_test)
    labels = model.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (model, metrics.classification_report(y_test, labels)))


if __name__ == '__main__':
    # main2()
    # search_parameters()
    # search_parameters2()
    train_use_searched_parameter()
    # load_model()
    # AdaBoostClassifier.predict_proba()
    # RandomForestClassifier.predict_proba()
    # MLPClassifier.predict_proba()
    # ExtraTreesClassifier.predict_proba()
    # DecisionTreeClassifier.predict_proba()
    # KNeighborsClassifier.predict_proba()