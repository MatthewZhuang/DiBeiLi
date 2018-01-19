#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
 @description:
        
 @Time       : 17/7/4 下午10:47
 @Author     : guomianzhuang
"""
import preprocess as pre
from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LogisticRegression


def ensemble_voting_weighted():
    # models = ["AdaBoostClassifier.m", "DecisionTreeClassifier.m",
    #           "ExtraTreesClassifier.m", "KNN.m", "MLPClassifier.m",
    #           "RandomForestClassifier.m", "SVC_rbf.m", "VotingClassifier.m"]
    # weights = [0.74, 0.76, 0.80, 0.76, 0.75, 0.81, 0.75, 0.72]
    models = ["AdaBoostClassifier.m", "DecisionTreeClassifier.m",
              "ExtraTreesClassifier.m", "KNN.m", "MLPClassifier.m",
              "VotingClassifier.m"]
    weights = [0.74, 0.76, 0.80, 0.76, 0.75, 0.72]
    data = pre.load_data(train=False)
    data = pre.process_without_discretize(data)
    X, X_test, y, y_test = pre.generate_corpus_for_continues(data, test_size=0.9999)
    labels = []
    data = pd.DataFrame(columns=models)
    for m in models:
        model = joblib.load("../models/" + m)
        label = model.predict(X_test)
        labels.append(label)
        data[m] = label

    print data.corr()
    rs_label = []
    for i in range(len(y_test)):
        w_sum = 0
        for j in range(len(labels)):
            l = labels[j][i]
            if l == 1:
                # w_sum += weights[j]
                w_sum += 1
            else:
                # w_sum += -1*weights[j]
                w_sum += -1
        if w_sum >= 0:
            rs_label.append(1)
        else:
            rs_label.append(0)

    print("Classification report for classifier %s:\n%s\n"
          % (model, metrics.classification_report(y_test, rs_label)))


def ensemble_blending():
    # models = ["AdaBoostClassifier.m", "DecisionTreeClassifier.m",
    #           "ExtraTreesClassifier.m", "KNN.m",
    #           "RandomForestClassifier.m"]
    # weights = [0.74, 0.76, 0.80, 0.76, 0.81]

    models = ["ExtraTreesClassifier.m",
              "RandomForestClassifier.m"]

    weights = [0.80, 0.81]

    # train data
    data = pre.load_data(train=True)
    data = pre.process_without_discretize(data)
    X_train, X, y_train, y = pre.generate_corpus_for_continues(data, test_size=0.0001)
    # test data
    data = pre.load_data(train=False)
    data = pre.process_without_discretize(data)
    X, X_test, y, y_test = pre.generate_corpus_for_continues(data, test_size=0.9999)
    print len(X_train)
    trains = []
    probs = []
    tests = []
    probs_tests = []
    for m in models:
        model = joblib.load("../models/" + m)
        prob_tmp = model.predict_proba(X_train)
        prob = []
        for i in range(len(prob_tmp)):
            prob.append(prob_tmp[i][1])
        probs.append(prob)

        prob_tmp = model.predict_proba(X_test)
        prob = []
        for i in range(len(prob_tmp)):
            prob.append(prob_tmp[i][1])
        probs_tests.append(prob)



    for i in range(len(X_train)):
        train = []
        for j in range(len(models)):
            train.append(probs[j][i])
        trains.append(train)

    for i in range(len(X_test)):
        test = []
        for j in range(len(models)):
            test.append(probs_tests[j][i])
        tests.append(test)

    clf = LogisticRegression()
    clf.fit(trains, y_train)
    rs_label = clf.predict(tests)


    print("Classification report for classifier %s:\n%s\n"
          % (model, metrics.classification_report(y_test, rs_label)))


if __name__ == '__main__':
    # ensemble_voting_weighted()
    ensemble_blending()