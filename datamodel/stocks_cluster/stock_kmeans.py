# -*- coding: utf-8 -*-
__author__ = 'Mike'
import os
import numpy as np
from matplotlib import pyplot as plt
# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras import layers
import pandas as pd
from keras.initializers import glorot_uniform
# import keras.utils.generic_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


data1 = pd.read_csv('../prepare_data/res/410030031937/train_data.csv')
data2 = pd.read_csv('../prepare_data/res/410001005971/train_data.csv')
raw_data = pd.concat([data1, data2], ignore_index=True)
print(raw_data.columns.to_list())
raw_data = raw_data.drop(['open', 'close', 'high', 'low', 'vol','stockname','selltime','sell_order','days_diff'], axis=1)
print(raw_data.columns.to_list())
#1、根据订单，将数据进行均值整合
agg_map = {}
for op in raw_data.columns.to_list():
    if op in ['min_prc', 'min_vol']:
        agg_map[op] = 'mean'
    else:
        agg_map[op] = 'first'

group_data = raw_data.groupby(['stockid', 'buy_order']).agg(agg_map).reset_index(drop=True)#.drop(['buy_order','stockid'], axis=1)
pd.set_option('display.max_columns', None)  # 显示所有列
# print(group_data.head(5))
group_data.to_csv('group_data.csv', sep=',')
group_data = group_data.drop(['buytime', 'buy_volume', 'sell_volume', 'timestamp', 'buy_order'], axis=1)
#['stockid', 'buytime', 'profit', 'buy_price', 'sell_price', 'buy_volume', 'sell_volume',
# 'buy_order', 'trade_diff', 'liutongguben', 'day_open_0', 'day_open_1',
# 'day_open_2', 'day_open_3', 'day_close_0', 'day_close_1', 'day_close_2', 'day_close_3', 'day_high_0', 'day_high_1',
# 'day_high_2', 'day_high_3', 'day_low_0', 'day_low_1', 'day_low_2', 'day_low_3', 'day_vol_0', 'day_vol_1', 'day_vol_2',
# 'day_vol_3', 'day_hsl_0', 'day_hsl_1', 'day_hsl_2', 'day_hsl_3', 'min_prc', 'min_vol', 'timestamp']
filtered = group_data[group_data.isna().any(axis=1)]
print(filtered)
group_data = group_data.fillna(0)

print(group_data.columns.to_list())

y = group_data['stockid']
X = group_data.drop('stockid', axis=1)
X = X.astype('float64')
print(X.columns.to_list())
print(X.describe())


kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 获取聚类中心和分配的标签
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# 可视化聚类结果
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50, alpha=0.7)
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
# plt.title('K-Means Clustering')
# plt.legend()
# plt.show()
