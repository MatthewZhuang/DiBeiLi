#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 7/3/2017 10:45 AM
    @desc:
        离散变量
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

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


def main():
    logging.info("stage: main method...")
    data = pre.load_data()
    data = pre.process(data)
    X_train, X_test, y_train, y_test = pre.generate_corpus(data, test_size=0.2)
    logging.info("stage: searching begin...")
    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', RandomForestClassifier(random_state=42))
    ])
    N_FEATURES_OPTIONS = [200, 250]
    n_estimators = [80, 100, 120]
    max_depth = [30, 40, 50]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7)],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__n_estimators': n_estimators,
            'classify__max_depth': max_depth
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__n_estimators': n_estimators,
            'classify__max_depth': max_depth
        },
    ]
    reducer_labels = ['PCA', 'KBest(chi2)']
    grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
    grid.fit(X_train, y_train)
    logging.info("stage: searching end...")

    print(grid.best_params_)
    estimator = grid.best_estimator_
    labels = estimator.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (estimator, metrics.classification_report(y_test, labels)))
    joblib.dump(estimator, "../models_lisan/RandomForestClassifier2.m")


def predict_with_trained():
    logging.info("stage: method predict_with_trained...")
    data = pre.load_data()
    data = pre.process(data)
    X_train, X_test, y_train, y_test = pre.generate_corpus(data, test_size=0.9)
    # model1
    model = joblib.load("../models_lisan/RandomForestClassifier.m")
    labels = model.predict(X_test)

    # for x, y in zip(X_test, y_test):
    #     y_ = model.predict(x)
    #     print x
    #     if y_ == y:
    #         print str(y_) + "----" + str(y)

    print("Classification report for classifier %s:\n%s\n"
          % (model, metrics.classification_report(y_test, labels)))

if __name__ == '__main__':
    main()
    # predict_with_trained()
    """
        {'classify__C': 0.1, 'reduce_dim': PCA(copy=True, iterated_power=7, n_components=350, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False), 'reduce_dim__n_components': 350}
Classification report for classifier LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0):
             precision    recall  f1-score   support

        0.0       0.72      0.65      0.68      1578
        1.0       0.76      0.81      0.79      2191

avg / total       0.74      0.75      0.74      3769

    """