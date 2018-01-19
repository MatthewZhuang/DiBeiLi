#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 7/3/2017 10:45 AM
    @desc:

    @author: guomianzhuang
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt


def load_data(train=True):
    if train:
        path = "/Users/Matthew/Documents/workspace/project/train.csv"
    else:
        path = "/Users/Matthew/Documents/workspace/project/test.csv"
    data = pd.DataFrame.from_csv(path)
    data.columns = ["ticker", "m_60", "dist_low", "dif_diff", "m_20",
                    "low_2", "low_1", "surpass", "down", "sz20_pos"
                    , "hs_m_20", "hs60_pos", "sz_m_60", "hs20_pos", "sz_m_20"
                    , "sz60_pos", "hs_m_60", "buy_60_pos", "buy_20_pos",
                    "date", "growth_rate", "cal_market_cap", "30_avg_growth_rate"]
    # std = data["30_avg_growth_rate"].std()
    # mean = data["30_avg_growth_rate"].mean()
    # data = data[data["30_avg_growth_rate"] > (mean - 3*std)]
    # data = data[data["30_avg_growth_rate"] < (mean + 3*std)]
    data.reset_index(inplace=True)
    label = []
    for i in range(len(data)):
        if data.iloc[i, 23] >= 0:
            label.append(1)
        else:
            label.append(0)
    data["label"] = label
    # print data
    # data["30_avg_growth_rate"].hist(bins=80).plot()
    # plt.show()
    del data["ticker"]
    del data["date"]
    del data["index"]
    return data


def transform_market_cap(market_cap):
    """
        转化市值
    """
    if market_cap < 0:
        return 0
    if market_cap < 2:
        return 1
    elif market_cap < 5:
        return 2
    elif market_cap < 10:
        return 3
    elif market_cap < 15:
        return 4
    elif market_cap < 20:
        return 5
    elif market_cap < 30:
        return 6
    elif market_cap < 40:
        return 7
    elif market_cap < 50:
        return 8
    elif market_cap < 70:
        return 9
    elif market_cap < 100:
        return 10
    elif market_cap < 150:
        return 11
    elif market_cap < 300:
        return 12
    elif market_cap < 500:
        return 13
    else:
        return 14


def normalization(value, mean, std):
    return (value - mean)/std


def process(data):
    logging.info("stage: processing the data...")
    raw_len = len(data)
    data.dropna(inplace=True)
    # 过滤异常数据
    data = data[data["30_avg_growth_rate"] < 100]
    data = data[data["growth_rate"] < 100]
    data = data[data["m_20"] < 100]
    new_len = len(data)
    print "fitered:" + str(raw_len - new_len)
    # 消除噪声数据
    data = data[(data["30_avg_growth_rate"] > 1) | (data["30_avg_growth_rate"] < -1)]
    # 各维度单独进行归一化 如果执行离散化，没必要进行归一化
    # cols = list(data.columns)
    # for i in range(22):
    #     if i in [6, 19, 20, 21]:
    #         continue
    #     col = cols[i]
    #     mean = data[col].mean()
    #     std = data[col].std()
    #     data[col] = data[col].apply(normalization, args=(mean, std))

    # 数据离散化
    data = discretize(data)
    return data


def process_without_discretize(data):
    logging.info("stage: method process_without_discretize...")
    raw_len = len(data)
    data.dropna(inplace=True)
    # 过滤异常数据
    data = data[data["30_avg_growth_rate"] < 100]
    data = data[data["growth_rate"] < 100]
    data = data[data["m_20"] < 100]
    new_len = len(data)
    print "fitered:" + str(raw_len - new_len)
    # 消除噪声数据
    data = data[(data["30_avg_growth_rate"] > 1) | (data["30_avg_growth_rate"] < -1)]
    # 各维度单独进行归一化 如果执行离散化，没必要进行归一化
    cols = list(data.columns)
    for i in range(22):
        if i in [6, 19, 20, 21]:
            continue
        col = cols[i]
        mean = data[col].mean()
        std = data[col].std()
        data[col] = data[col].apply(normalization, args=(mean, std))

    # 数据离散化
    data["cal_market_cap"] = data["cal_market_cap"].apply(transform_market_cap)
    data = data.reset_index(drop=True)
    print cols
    return data


def discretize(data):
    """
        进行数据离散化
        根据值排序，每一千取一个样本点
        1、等概率划分区间 selected
        2、分位数划分    bad
        3、根据经验划分 best (eg:市值)
    """
    logging.info("stage: discretize the data...")
    cols = list(data.columns)
    cols_index = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    for col in cols_index:
        data.sort_values(by=cols[col], inplace=True)
        data = data.reset_index(drop=True)
        label = 0
        for i in range(len(data)):
            if i % 1000 == 0:
                label += 1
            data.iloc[i, col] = label
    data["cal_market_cap"] = data["cal_market_cap"].apply(transform_market_cap)
    return data


def generate_corpus_for_continues(data, test_size=0.2):
    """
        部分特征离散化
    """
    logging.info("stage: method generate_corpus_for_continues...")
    from sklearn.model_selection import train_test_split
    train = []
    label = []
    for i in range(len(data)):
        line = data.loc[i]
        record = list(line[0:19])
        tag = line[19]
        market_cap = [0] * 14
        market_cap[int(tag-1)] = 1
        record.extend(market_cap)
        train.append(record)
        label.append(line[21])
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=test_size)
    return X_train, X_test, y_train, y_test


def generate_corpus(data, test_size=0.2):
    """
        特征离散化  全部转化为01模型
    """
    logging.info("stage: generate the corpus...")
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    encoder = OneHotEncoder()
    train = []
    label = []
    for i in range(len(data)):
        line = data.loc[i]
        train.append(line[0:20])
        label.append(line[21])
    encoder.fit(train)
    train = encoder.transform(train).toarray()
    print "dimensions:" + str(len(train[0]))
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=test_size)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data = load_data()
    data = process(data)
    train_x, train_y, test_x, test_y = generate_corpus(data)

