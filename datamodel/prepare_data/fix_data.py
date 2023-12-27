# -*- coding: utf-8 -*-
__author__ = 'Mike'
from pytdx.hq import TdxHq_API
import pandas as pd
from datetime import datetime
from io import StringIO

current_datetime = datetime.now()
current_date = current_datetime.date().strftime('%Y-%m-%d')

api = TdxHq_API()
with api.connect('119.147.212.81', 7709):
    trade_dates = api.get_k_data('000001','2023-02-20',current_date).index.tolist()
print(trade_dates)

def fix_day_info(code, buy_date):
    start_date = trade_dates[trade_dates.index(buy_date) - 3]
    with api.connect('119.147.212.81', 7709):
        k_df = api.get_k_data(code, start_date, buy_date)
    features = ['open', 'close', 'high', 'low', 'vol']
    feature_data = {}
    for f in features:
        feature_data[f] = k_df[f].tolist()
    return feature_data

# 定义一个函数用于转换字符串为数组
def convert_to_list(s):
    return eval(s)

df = pd.read_csv("/home/zhaomengyi/Projects/QuantProjects/QuantDataMiner/datamodel/train_data.csv")
# df['open'] = df['open'].apply(lambda x : eval(x))
pd.set_option('display.max_columns', None)  # 显示所有列
#先过滤出NaN的
filter_df = df[df['close'].apply(eval).apply(lambda x : x is None or type(x) is float or len(list(x))==0)]
stocks = filter_df['stockid'].to_list()
buytimes = filter_df['buytime'].to_list()
result = dict(zip(stocks, buytimes))
print(result)
# ['SH.600391', 'SH.600415', 'SH.600629', 'SH.600636', 'SH.600892', 'SH.600961', 'SH.603016', 'SH.603103', 'SH.603215', 'SH.603598', 'SH.603738', 'SZ.000403', 'SZ.000409', 'SZ.000663', 'SZ.000802', 'SZ.002174', 'SZ.002215', 'SZ.002253', 'SZ.002261', 'SZ.002281', 'SZ.002463', 'SZ.002530', 'SZ.002558', 'SZ.002600', 'SZ.002681', 'SZ.300052', 'SZ.300078', 'SZ.300128', 'SZ.300161', 'SZ.300195', 'SZ.300251', 'SZ.300264', 'SZ.300419', 'SZ.300460', 'SZ.300462', 'SZ.300616', 'SZ.300654', 'SZ.300807', 'SZ.300922', 'SZ.300949']

for key, value in result.items():
    #fix data
    try:
        code = key.split('.')[1]
        buydate = value.split(' ')[0]
        print(key, buydate, fix_day_info(code, buydate))
    except:
        print(key, buydate)
