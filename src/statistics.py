#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 7/4/2017 3:08 PM
    @desc:
        
    @author: guomianzhuang
"""
import matplotlib.pyplot as plt


def statistics(data):
    # data.sort_values(by="cal_market_cap", inplace=True)
    # vs = data["cal_market_cap"].values
    # print vs
    # print data
    data["cal_market_cap"].hist(bins=14).plot()
    plt.show()


if __name__ == '__main__':
    import preprocess as pre
    data = pre.load_data()
    data = pre.process(data)
