# -*- coding: utf-8 -*-
__author__ = 'Mike'
import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense
from pytdx.hq import TdxHq_API
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta
from retrying import retry
from ast import literal_eval
import ast
import collections
import numpy as np
import json

current_datetime = datetime.now()
current_date = current_datetime.date().strftime('%Y-%m-%d')

api = TdxHq_API()
with api.connect('119.147.212.81', 7709):
    trade_dates = api.get_k_data('000001','2023-01-20',current_date).index.tolist()
print(trade_dates)

def trading_days_diff(row):
    selltime = row['selltime'].split(' ')[0]
    buytime = row['buytime'].split(' ')[0]
    end = trade_dates.index(selltime)
    start = trade_dates.index(buytime)
    return end - start

# @retry 装饰器用于修饰 get_data 函数，它会在请求失败时触发重试，最多重试 3 次（stop_max_attempt_number=3）。
# wait_fixed 参数表示每次重试之间的等待时间（这里设定为 1000 毫秒，即 1 秒）。
@retry(wait_fixed=1000, stop_max_attempt_number=3)
def get_history_min_data(row):
    stock_id = row['stockid']
    buy_time = row['buytime']
    buy_date = buy_time.split(' ')[0].replace('-','')
    market = get_market(stock_id.split('.')[0])
    code = stock_id.split('.')[1]
    with api.connect('119.147.212.81', 7709):
        data = api.get_history_minute_time_data(market, code, buy_date)
    return data


#DNN处理方法，在里面求平均值
def split_min_data_DNN(row, tag, start_index, end_index):
    min_info = row['min_info']
    if min_info ==  None:
        print(row['stockid'],row['buytime'])
        return
    #技巧：将一列元素的一部分数据转换成一列
    avg = 0
    try:
        min_info = min_info.replace('OrderedDict', 'collections.OrderedDict')
        min_info_arr = eval(min_info)
        min_info_get = min_info_arr[start_index:end_index + 1]
        res_arr = [ele[tag] for ele in min_info_get]
        avg = round(np.mean(res_arr), 2)
    except:
        print(row._name, row['stockid'],row['buytime'])
    return avg

#RNN方法，对数据进行保留
def split_min_data_RNN(row, tag):
    min_info = row['min_info']
    if min_info == None:
        print(row['stockid'], row['buytime'])
        return
    try:
        min_info = min_info.replace('OrderedDict', 'collections.OrderedDict')
        min_info_arr = eval(min_info)
        res_arr = [ele[tag] for ele in min_info_arr]
    except:
        print(row._name, row['stockid'], row['buytime'])
    return res_arr


def make_timestamp(row):
    buy_date = row['buytime'].split(' ')[0]
    start_time = datetime.strptime(buy_date + " 09:30:00", '%Y-%m-%d %H:%M:%S')
    min_prc = row['min_prc']
    min_prcs_cnt = len(min_prc)
    timestamps = [(start_time + timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(min_prcs_cnt)]
    return timestamps



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


def get_market(tag):
    if tag=='SZ':
        return 0
    elif tag=='SH':
        return 1
    else:
        return -1

# 读取 CSV 文件时指定 Ordered Dict 列的转换函数
# 读取 CSV 文件时，使用适当的解析函数还原 Ordered Dict 列
def parse_ordered_dict_array(string):
    try:
        # 使用 ast.literal_eval 解析字符串
        return ast.literal_eval(string.replace('OrderedDict', 'collections.OrderedDict'))
    except (ValueError, SyntaxError):
        return string

# 自定义解析函数，将 JSON 字符串转换为字典数组
def parse_json_array(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return []


# history = get_history_min_data('SZ.002494', '2023-12-13 09:25:00')
# print(history[0]['price'])
# print(history[1]['vol'])

#1、读取mysql的原始数据，增加分钟数据
''''
data = pd.read_csv('/home/zhaomengyi/Projects/QuantProjects/QuantDataMiner/datamodel/prepare_data/res/410001005971/410001005971.csv')
print(data.columns)
data['trade_diff'] = data.apply(lambda row: trading_days_diff(row), axis=1)
data['min_info'] = data.apply(lambda row: get_history_min_data(row), axis=1)
print(data[['trade_diff', 'days_diff', 'min_info']])
data.to_csv('data_min.csv', index=False, sep=',', header=True)
'''
'''
#2、在分钟数据上添加日线数据和财务数据
data = pd.read_csv('/home/zhaomengyi/Projects/QuantProjects/QuantDataMiner/datamodel/prepare_data/res/410001005971/data_min.csv')
data['day_info'] = data.apply(lambda row: get_history_day_data(row), axis=1)
data['liutongguben'] = data.apply(lambda row:get_finance_data(row, 'liutongguben'), axis=1)
data.to_csv('data_full_info.csv', index=False, sep=',', header=True)
'''
data = pd.read_csv('/home/zhaomengyi/Projects/QuantProjects/QuantDataMiner/datamodel/prepare_data/res/410030031937/data_full_info.csv',
                   escapechar=None, converters={'min_info':parse_ordered_dict_array})
# data['min_info'] = data['min_info'].apply(parse_ordered_dict_array)




#3_1data['day_info']的数据类型时字典，key为日线上各种指标，value是近四天指标的具体数值
data['day_info'] = data['day_info'].apply(literal_eval).apply(dict)
#技巧：将字典拆成多列，以day_vol为例子，拆成day_vol_0道day_vol_3
data = data['day_info'].apply(pd.Series).merge(data, left_index=True, right_index=True)
#生成对应日线,'open', 'close', 'high', 'low', 'vol'
data[['day_open_' + str(i) for i in range(4)]] = data['open'].apply(pd.Series)
data[['day_close_' + str(i) for i in range(4)]] = data['close'].apply(pd.Series)
data[['day_high_' + str(i) for i in range(4)]] = data['high'].apply(pd.Series)
data[['day_low_' + str(i) for i in range(4)]] = data['low'].apply(pd.Series)
data[['day_vol_' + str(i) for i in range(4)]] = data['vol'].apply(pd.Series)
#最后求一些换手率
for i in range(4):
    data['day_hsl_' + str(i)] =  data['day_vol_' + str(i)] / data['liutongguben']

#3_2方法1：将min_info分成8组，每3个一组,DNN处理方法，把一天的分钟数据都打在8条八小时数据中
'''
for i in range(8):
    prc_tag = 'min_prc_' + str(i)
    vol_tag = 'min_vol_' + str(i)
    start_index = i * 30
    end_index = i * 30 + 29
    data[prc_tag] = data.apply(lambda row:split_min_data_DNN(row, 'price', start_index, end_index), axis=1)
    data[vol_tag] = data.apply(lambda row:split_min_data_DNN(row, 'vol', start_index, end_index), axis=1)
'''
#3_2方法2：RRN时序处理方法，将240个分钟直接暴力打散
data['min_prc'] = data.apply(lambda row:split_min_data_RNN(row, 'price'), axis=1)
data['min_vol'] = data.apply(lambda row:split_min_data_RNN(row, 'vol'), axis=1)
data['timestamp'] = data.apply(lambda row:make_timestamp(row), axis=1)
data = data.explode(['min_prc', 'min_vol', 'timestamp'])




data = data.drop(['day_info','min_info'], axis=1)
pd.set_option('display.max_columns', None)  # 显示所有列
data.to_csv('train_data.csv', index=False, sep=',', header=True)
