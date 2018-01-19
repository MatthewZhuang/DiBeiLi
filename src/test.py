#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 7/3/2017 1:21 PM
    @desc:
        
    @author: guomianzhuang
"""
import pandas as pd

# def normalization(value, mean, std):
#     return (value - mean)/std


# data = pd.DataFrame(data=[[1, 2, 3], [2, 3, 7]], columns=list("ABC"))
# print data
# data["B"] = data["B"].apply(normalization, args=(1, 1))
# print data
# print list(data.loc[0])
# t = [1]
# tag =  [0]*14
# print t.extend(tag)
# print t


path = "/Users/Matthew/Documents/workspace/project/dibeili_20060101_20170701_v.2.csv"
data = pd.DataFrame.from_csv(path)
print len(data)
data = data.reset_index()
# print data
import numpy as np


def shuffle(df, n=1, axis=0):
    print df.index
    df = df.copy()
    df = df.reset_index()
    for _ in range(n):
        df = df.reindex(np.random.permutation(df.index))
        print df.iloc[0, 0]
        # df = df.reset_index()
        # df.drop("index", inplace=True)
    df = df.reset_index(drop=True)
    del df['index']
    del df['1']
    return df

data = shuffle(data, n=10)
print data
d1 = data[:4000]
d2 = data[4000:]
# print d1
d1.to_csv("/Users/Matthew/Documents/workspace/project/test.csv")
d2.to_csv("/Users/Matthew/Documents/workspace/project/train.csv")



# def t1():
#     import preprocess as pre
#     data = pre.load_data()
#     print data
#
#
# if __name__ == '__main__':
#     t1()
