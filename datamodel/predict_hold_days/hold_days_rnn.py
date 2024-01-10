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
import joblib



sampling_rate = 6
sequence_length = 15#240#100#240#120#120
delay = sampling_rate * (sequence_length + 24 -1)#239#sampling_rate * (sequence_length + 24 -1) #239 #
batch_size = 10#300 #256

data1 = pd.read_csv('../prepare_data/res/410030031937/train_data.csv')
data2 = pd.read_csv('../prepare_data/res/410001005971/train_data.csv')
raw_data = pd.concat([data1, data2], ignore_index=True)
raw_data = raw_data.fillna(0)
print(raw_data.columns.to_list())
raw_data = raw_data[raw_data['buy_price'] < 20]
raw_data = raw_data.drop(['open', 'close', 'high', 'low', 'vol', 'stockname', 'selltime', 'days_diff','buytime','stockid',
                  'buy_volume', 'sell_volume', 'sell_order','profit','sell_price'
                          # 'liutongguben',
                          # , 'day_vol_3','day_hsl_0','min_vol', 'day_open_1', 'day_open_2', 'day_open_0','day_close_0','day_close_1',
                          # 'day_close_2',
                          # 'day_high_0', 'day_high_1', 'day_high_2', 'day_vol_0','day_vol_1','day_vol_2','day_vol_3', 'day_hsl_0',
                          # 'day_hsl_1','day_hsl_2', 'day_hsl_3'
                          ], axis=1) #'buy_order','timestamp',
# raw_data = raw_data[raw_data['trade_diff'] < 100]
raw_data = raw_data.sort_values(['buy_order', 'timestamp'])
raw_data = raw_data.drop(['buy_order', 'timestamp'], axis=1)




num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_val_samples - num_train_samples

target = raw_data['trade_diff'].astype(float)
# print(target)
raw_data = raw_data.drop('trade_diff', axis=1)
#替换成0到1区间
scaler = MinMaxScaler(feature_range=(0,1))
raw_data = pd.DataFrame(scaler.fit_transform(raw_data), columns=raw_data.columns)
joblib.dump(scaler, 'rawdata_minmax_scaler.pkl')
print(raw_data.columns.to_list)

#处理targets
predict_scaler = MinMaxScaler(feature_range=(0,1))
reshape_target = target.values.reshape(-1, 1)
target = predict_scaler.fit_transform(reshape_target)
joblib.dump(predict_scaler, 'predict_minmax_scaler.pkl')
print(target)


statics = pd.DataFrame(raw_data).describe()
print(statics)
statics.to_csv('statics.csv')



# print(raw_data.columns.tolist())



train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=target[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True, #让256个批次的数值乱序
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=target[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples
)
test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=target[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples
)

for samples, targets in train_dataset:#迭代每一批次的样本
    print("sample shape", samples.shape) #(256,120,14) 理解：256个批量样本中，每个样本中有120个时间序列样本，每个单位时间的样本有14个维度特征
    print("target shape", targets.shape) #(256,)
    break





inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
#技巧在使用cpu训练模型的时候，可以把这里进行展开unroll=true
# x = layers.Dense(units=16, kernel_initializer=glorot_uniform(seed=42))(inputs)
x = layers.LSTM(64, recurrent_dropout=0.5, activation='relu', return_sequences=True)(inputs) #recurrent_dropout=0.25,
x = layers.LSTM(64, return_sequences=False)(x)
x = layers.Dense(32)(x)
# x = layers.LSTM(64, return_sequences=False)(x)
# x = layers.Dense(25)(x)
# x = layers.Dropout(0.25)(x) #进行正则化
outputs = layers.Dense(1)(x)
# outputs = layers.Dense(units=1, activation='sigmoid')(x) #试试二分类
model = keras.Model(inputs, outputs)
callbacks = [keras.callbacks.ModelCheckpoint("hold_days.keras", save_best_only=True)]
optimizer = keras.optimizers.RMSprop(learning_rate=1e-3) #epsilon=1e-4, clipvalue=0.2) #, clipvalue=0.5梯度截断
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
print(model.summary())

history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=callbacks)

loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Val MAE")
plt.title("Training and Val MAE")
plt.legend()
plt.show()
plt.savefig("hold_days_LSTM_MAE.png")