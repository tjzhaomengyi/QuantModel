# -*- coding: utf-8 -*-
__author__ = 'Mike'
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import pandas as pd
from keras.initializers import glorot_uniform
# import keras.utils.generic_utils
from sklearn.preprocessing import MinMaxScaler
import configparser
import mysql.connector
import joblib
from pytdx.hq import TdxHq_API
from datetime import datetime, timedelta
from retrying import retry
from ast import literal_eval
import math





config = configparser.ConfigParser()
config.read('../../configs/config.ini')
mysql_config = {
    'host' : config['mysql']['host'],
    'user' : config['mysql']['user'],
    'password' : config['mysql']['password'],
    'database' : config['mysql']['database'],
    'port' : config['mysql']['port']
}


current_datetime = datetime.now()
current_date = current_datetime.date().strftime('%Y-%m-%d')

api = TdxHq_API()
with api.connect('119.147.212.81', 7709):
    trade_dates = api.get_k_data('000001','2023-01-20',current_date).index.tolist()
# print(trade_dates)

def get_stock_buy(buy_date):
    try:
        connection = mysql.connector.connect(**mysql_config)
        if connection.is_connected():
            print('Connectoed to MySql database')
            cursor = connection.cursor()
            select_sql = f"select stockid, price, tradetime from quant.deal where tradetime like '{buy_date}%' and offsetflagtype='48' group by stockid"
            df = pd.read_sql_query(select_sql, connection)
            return df
    except Exception as e:
        print(f'Error: {e}')
    finally:
        #关闭数据库
        if connection.is_connected():
            connection.close()
            print('connenction closed')


def get_market(tag):
    if tag=='SZ':
        return 0
    elif tag=='SH':
        return 1
    else:
        return -1

def make_timestamp(row):
    buy_date = str(row['buytime']).split(' ')[0]
    start_time = datetime.strptime(buy_date + " 09:30:00", '%Y-%m-%d %H:%M:%S')
    min_prc = row['min_prc']
    min_prcs_cnt = len(min_prc)
    timestamps = [(start_time + timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(min_prcs_cnt)]
    return timestamps

# @retry 装饰器用于修饰 get_data 函数，它会在请求失败时触发重试，最多重试 3 次（stop_max_attempt_number=3）。
# wait_fixed 参数表示每次重试之间的等待时间（这里设定为 1000 毫秒，即 1 秒）。
@retry(wait_fixed=1000, stop_max_attempt_number=3)
def get_history_min_data(row):
    stock_id = row['stockid']
    buy_time = str(row['buytime'])
    buy_date = buy_time.split(' ')[0].replace('-','')
    market = get_market(stock_id.split('.')[0])
    code = stock_id.split('.')[1]
    with api.connect('119.147.212.81', 7709):
        data = api.get_history_minute_time_data(market, code, buy_date)
        # data = api.get_security_bars(7, market, code, 1, 240)
    return data


@retry(wait_fixed=1000, stop_max_attempt_number=3)
def get_history_day_data(row):
    if (row._name + 1) % 50 == 0:
        print("日线数据拉取数据到：" + str(row._name) + "行")
    code = row['stockid'].split('.')[1]
    buy_date = str(row['buytime']).split(' ')[0]
    start_date = trade_dates[trade_dates.index(buy_date) - 3]
    feature_data = {}
    with api.connect('119.147.212.81', 7709):
        try:
            k_df = api.get_k_data(code, start_date, buy_date)
            features = ['open', 'close', 'high', 'low', 'vol']
            for f in features:
                feature_data[f] = k_df[f].tolist()
        except Exception as e:
            print(f"{code},{start_date},{buy_date}")
            print(str(e))
    return feature_data




@retry(wait_fixed=1000, stop_max_attempt_number=3)
def get_finance_data(row, feature):
    code = row['stockid'].split('.')[1]
    market = get_market(row['stockid'].split('.')[0])
    with api.connect('119.147.212.81', 7709):
        finance = api.get_finance_info(market, code)
    if (row._name+1) % 50 == 0:
        print(feature + "拉取数据到：" + str(row._name) + "行")
    info = finance[feature]
    return info



#RNN方法，对数据进行保留
def split_min_data_RNN(row, tag):
    min_info = row['min_info']
    if min_info == None:
        print(row['stockid'], row['buytime'])
        return
    try:
        #注意：这里不是从csv中读取的字符串，所以不需要进行eval的字符串解析成对象的过程
        # min_info = min_info.replace('OrderedDict', 'collections.OrderedDict')
        # min_info_arr = eval(min_info)
        res_arr = [ele[tag] for ele in min_info]
    except:
        print(row._name, row['stockid'], row['buytime'])
    return res_arr

#创建时序数据集
def create_sequences(raw_data, targets, look_back):
    raw_data = raw_data.values
    targets = targets.values
    X, y = [], []
    for i in range(len(raw_data) - look_back):
        X.append(raw_data[i:(i+look_back), :])
        y.append(targets[i + look_back])
    return np.array(X), np.array(y)

def create_sequeces_predict(raw_data, look_back):
    raw_data = raw_data.values
    X= []
    for i in range(len(raw_data) - look_back):
        X.append(raw_data[i:(i + look_back), :])
    return np.array(X)

#
# def dataframe_to_array(df, look_back):
#     data = df.values
#     sequences, labels = create_sequence(data, look_back)
#     return sequences.reshape(-1, look_back, 1), labels

#为了让模型也加载规范化的数据，这里应该保存一下这个规范化更合适
# data1 = pd.read_csv('../prepare_data/res/410030031937/train_data.csv')
# data2 = pd.read_csv('../prepare_data/res/410001005971/train_data.csv')
# raw_data = pd.concat([data1, data2], ignore_index=True)
# raw_data = raw_data.fillna(0)
# print(raw_data.columns.to_list())
# raw_data = raw_data.drop(['open', 'close', 'high', 'low', 'vol', 'stockname', 'selltime', 'days_diff','buytime','stockid',
#                   'buy_volume', 'sell_volume', 'sell_order','profit',
#                           #'liutongguben',
#                           #'profit', 'day_vol_3','day_hsl_0','min_vol', 'day_open_1', 'day_open_2', 'day_open_0','day_close_0','day_close_1',
#                           'day_close_2',
#                           'day_high_0', 'day_high_1', 'day_high_2', 'day_vol_0','day_vol_1','day_vol_2','day_vol_3', 'day_hsl_0',
#                           'day_hsl_1','day_hsl_2', 'day_hsl_3'
#                           ], axis=1) #'buy_order','timestamp',
# raw_data = raw_data.sort_values(['buy_order', 'timestamp'])
# targets = raw_data['trade_diff']
# raw_data = raw_data.drop(['buy_order', 'timestamp','trade_diff'], axis=1)
#
# scaler = MinMaxScaler(feature_range=(0,1))
# raw_data = pd.DataFrame(scaler.fit_transform(raw_data), columns=raw_data.columns)
# joblib.dump(scaler, 'rawdata_minmax_scaler.pkl')
#
# #预测结果专用的规范器
# predict_scaler = MinMaxScaler(feature_range=(0,1))
# reshape_target = targets.values.reshape(-1, 1)
# scaler_target = predict_scaler.fit_transform(reshape_target)
# joblib.dump(predict_scaler, 'predict_minmax_scaler.pkl')




# hold_days_model = keras.models.load_model("../predict_hold_days/hold_days.keras")
'''
#前15个来预测
raw = raw_data.iloc[-240:]
targets = targets.iloc[-240:]
# print(type(raw))
sampling_rate = 6
sequence_length = 15#240#100#240#120#120
delay = sampling_rate * (sequence_length + 24 -1)
batch_size = 10
X, y = create_sequences(raw, targets, sequence_length)

raw_predict = hold_days_model.predict(X)
predict_hold_days= predict_scaler.inverse_transform(raw_predict.reshape(-1, 1)).astype(int)
predict_hold_days = predict_hold_days.reshape(1, -1)[0]
raw_days = targets.iloc[-len(predict_hold_days):].values
duibi = list(zip(predict_hold_days, raw_days))
print(duibi)
for i in range(len(duibi)):
    if (i + 1) % 240 == 0:
        print('--------------')
    print(duibi[i])
'''
# 加载规范化
feature_names_in_ = ['buy_price', 'liutongguben', 'day_open_0' ,'day_open_1', 'day_open_2',
 'day_open_3' ,'day_close_0' ,'day_close_1', 'day_close_2', 'day_close_3',
 'day_high_0', 'day_high_1' ,'day_high_2', 'day_high_3' ,'day_low_0',
 'day_low_1', 'day_low_2', 'day_low_3', 'day_vol_0' ,'day_vol_1', 'day_vol_2',
 'day_vol_3', 'day_hsl_0', 'day_hsl_1', 'day_hsl_2', 'day_hsl_3', 'min_prc',
 'min_vol']
feature_nparr = np.array(feature_names_in_)
scaler = joblib.load('rawdata_minmax_scaler.pkl')
scaler.feature_names_in_ = feature_nparr #这里保证序列特征序列顺序即可，不加也可以，保存的scaler没有特征列名
predict_scaler = joblib.load('predict_minmax_scaler.pkl')
df = get_stock_buy('2024-01-10')
if df.empty :
    print('当日没有买入')
else:
    df = df.rename(columns={'price':'buy_price','tradetime':'buytime'})
    df['min_info'] = df.apply(lambda row: get_history_min_data(row), axis=1)
    df['day_info'] = df.apply(lambda row: get_history_day_data(row), axis=1)
    df['liutongguben'] = df.apply(lambda row: get_finance_data(row, 'liutongguben'), axis=1)

    # 3_1data['day_info']的数据类型时字典，key为日线上各种指标，value是近四天指标的具体数值
    df['day_info'] = df['day_info'].apply(dict) #.apply(literal_eval)
    # 技巧：将字典拆成多列，以day_vol为例子，拆成day_vol_0道day_vol_3
    df = df['day_info'].apply(pd.Series).merge(df, left_index=True, right_index=True)
    # 生成对应日线,'open', 'close', 'high', 'low', 'vol'
    df[['day_open_' + str(i) for i in range(4)]] = df['open'].apply(pd.Series)
    df[['day_close_' + str(i) for i in range(4)]] = df['close'].apply(pd.Series)
    df[['day_high_' + str(i) for i in range(4)]] = df['high'].apply(pd.Series)
    df[['day_low_' + str(i) for i in range(4)]] = df['low'].apply(pd.Series)
    df[['day_vol_' + str(i) for i in range(4)]] = df['vol'].apply(pd.Series)
    # 最后求一些换手率
    for i in range(4):
        df['day_hsl_' + str(i)] = df['day_vol_' + str(i)] / df['liutongguben']

    df['min_prc'] = df.apply(lambda row: split_min_data_RNN(row, 'price'), axis=1)
    df['min_vol'] = df.apply(lambda row: split_min_data_RNN(row, 'vol'), axis=1)
    df['timestamp'] = df.apply(lambda row: make_timestamp(row), axis=1)
    df = df.explode(['min_prc', 'min_vol', 'timestamp'])

    df = df.drop(['day_info', 'min_info', 'open', 'close', 'high', 'low', 'vol'], axis=1)
    pd.set_option('display.max_columns', None)  # 显示所有列




    #加载模型
    hold_days_model = keras.models.load_model("hold_days.keras")
    print(hold_days_model.summary())

    #按照每只股票进行预测
    stocks = set(df['stockid'].tolist())
    for stock in stocks:
        data = df[df['stockid']==stock]
        # print(data)
        data = data.drop(['stockid', 'timestamp', 'buytime'], axis=1)
        # print(len(data.columns.to_list))
        # print(data.columns.to_list)
        data = pd.DataFrame(scaler.transform(data))#注意：技巧这里一定要用transform，使用原来scaler有的最大最小值来卡新值的范围, columns=data.columns
        X = create_sequeces_predict(data,15)
        raw_predict = hold_days_model.predict(X)
        predict_hold_days = predict_scaler.inverse_transform(raw_predict.reshape(-1, 1)).astype(int)
        predict_hold_days = predict_hold_days.reshape(1, -1)[0]
        print(stock + ',' + str(math.ceil(predict_hold_days.mean())))